import logging
import os
import pickle
import shutil
import socket
import struct  # noqa
import sys
import time
import typing as t
from enum import Enum
from functools import lru_cache
from pathlib import Path
from threading import Thread

import cv2 as cv
import cv2.typing as ct
import numpy as np
import pyspark.sql.functions as F  # noqa
import pyspark.sql.types as T
from dotenv import load_dotenv
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

load_dotenv()

H, W = 480, 640
CONTOUR_AREA = 500
BG_SUBSTRACTOR = cv.createBackgroundSubtractorKNN()
PROCESSED_FRAME_SCHEMA = T.StructType(
    [
        T.StructField("frame_no", T.IntegerType()),
        T.StructField("frame", T.BinaryType()),
        T.StructField("frame_shape", T.ArrayType(T.IntegerType())),
        T.StructField("frame_dtype", T.StringType()),
        T.StructField("contours", T.BinaryType()),
        T.StructField("contours_len", T.IntegerType()),
        T.StructField("contours_dtype", T.StringType()),
        T.StructField("is_movement_present", T.BooleanType()),
        T.StructField("unknown", T.StringType()),
    ]
)
SOURCE_FILE_SCHEMA = T.StructType(
    [
        T.StructField("path", T.StringType()),
        T.StructField("modificationTime", T.TimestampType()),
        T.StructField("length", T.LongType()),
        T.StructField("content", T.BinaryType()),
    ]
)
SOURCE_SOCKET_SCHEMA = T.StructType(
    [
        T.StructField("value", T.StringType()),
    ]
)


class SinkType(Enum):
    socket = 0
    file = 1


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    logger = logging.getLogger("App")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()


def get_source() -> t.Optional[cv.VideoCapture]:
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        logger.info("Failed to open video stream")
        return None

    return capture


def is_valid_contour_area(contour: ct.MatLike) -> bool:
    return cv.contourArea(contour) >= CONTOUR_AREA * 4


def resize_and_blur(frame: ct.MatLike) -> ct.MatLike:
    frame = cv.resize(frame, (W, H))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (21, 21), 0)
    return blur


def get_fg_mask(frame: ct.MatLike) -> ct.MatLike:
    # ref - https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    return BG_SUBSTRACTOR.apply(frame)


def get_contours(mask: ct.MatLike) -> t.Generator[ct.MatLike, None, None]:
    # As long as there is movement in the image, we can expect to find contours in the frame
    # This is due to the knn background substractor being used.
    fgmask_contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in fgmask_contours:
        if not is_valid_contour_area(cnt):
            continue

        yield cnt


def draw_rect(frame: ct.MatLike, contour: ct.MatLike):
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def get_video_stream(source) -> t.Generator[ct.MatLike, None, None]:
    """
    yields the original frame captured, the mask and the Sequence of Contours
    yield frame, fg_mask, contours
    """
    while True:
        ret, frame = source.read()
        if not ret or frame is None:
            source.release()
            break

        yield frame


def process_frame(frame: ct.MatLike) -> t.Tuple[ct.MatLike, ct.MatLike, t.List[ct.MatLike]]:
    """
    params: frame: the original frame where the rectangle will be draw
    returns: frame, fg_mask, contours

    Resizes and blurs given frame before finding the contours
    """
    blur = resize_and_blur(frame)
    fg_mask = get_fg_mask(blur)
    contours = list(get_contours(fg_mask))
    for contour in contours:
        draw_rect(frame, contour)

    return frame, fg_mask, contours


class DataGenStreamThread(Thread):
    def __init__(
        self,
        sink: SinkType,
        source: cv.VideoCapture,
        port: t.Optional[int] = None,
        daemon: bool = False,
        directory: t.Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        self.port = port
        self.source = source
        self.sink = sink
        self.directory = (Path() / directory) if directory else None
        super().__init__(*args, **kwargs, daemon=daemon)

    def send_frame(self, frame: ct.MatLike, client: socket.socket, frame_no=-1) -> bool:
        try:
            data = {"frame_no": frame_no, "frame": frame.tobytes(), "shape": frame.shape, "dtype": frame.dtype}
            pickled_data = pickle.dumps(data)
            # packed = struct.pack("Q", pickled_frame)
            client.sendall(pickled_data)
            return False
        except:  # noqa
            return True

    def _setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("localhost", self.port))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        logger.info(f"Socket Listening on port={self.port!r}")
        self.start()
        return self.port

    def _setup_dir(self):
        if self.directory.exists():
            shutil.rmtree(self.directory.as_posix())

        self.directory.mkdir()

    def start_gen(self):
        if self.sink == SinkType.file:
            self._setup_dir()
        elif self.sink == SinkType.socket:
            self._setup_socket()
        else:
            raise NotImplementedError()

        self.start()

    def _handle_socket(self):
        while True:
            logger.info("Starting socket thread, going to accept")
            (client, addr) = self.sock.accept()
            logger.info(f"Client Connected {addr}")
            iters = 100
            for frame_no, frame in enumerate(get_video_stream(self.source)):
                if frame_no > iters:
                    logger.warning("Stopping data stream due to max iterations!")
                    break
                logger.info(f"Processing {frame_no=}")
                rv = self.send_frame(frame, client, frame_no=frame_no)
                time.sleep(0.5)
                if rv:
                    break

            logger.info("Socket Exiting Client Loop")
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                client.close()

    def _handle_file(self):
        iters = 100
        for frame_no, frame in enumerate(get_video_stream(self.source)):
            if frame_no > iters:
                logger.warning("Stopping data stream due to max iterations!")
                break
            time.sleep(0.5)
            logger.info(f"Processing {frame_no=}")
            data = {"frame_no": frame_no, "frame": frame.tobytes(), "shape": frame.shape, "dtype": frame.dtype}
            pickled_data = pickle.dumps(data)

            assert (
                self.directory is not None and self.directory.exists()
            ), "Directory for streaming the data files is not set"

            path = self.directory.absolute() / f"{frame_no}_data.pickled"
            with open(path.as_posix(), mode="wb") as f:
                f.write(pickled_data)

    def run(self) -> None:
        """Overrides Thread.run

        Creates a socket and waits(blocking) for connections
        When a connection is closed, goes back into waiting.
        """
        if self.sink == SinkType.socket:
            return self._handle_socket()
        elif self.sink == SinkType.file:
            return self._handle_file()

        raise NotImplementedError()

    def start(self):
        """Starts the socket thread"""
        Thread.start(self)


def basic_processing(read_dir=False):
    def f_sort(p: str) -> int:
        fname = os.path.basename(p)
        pref = fname.split("_")[0]
        return int(pref)

    if read_dir:
        import glob

        fs = glob.glob("data/*")
        fs.sort(key=f_sort)
        for f in fs:
            with open(f, "rb") as _f:
                data = pickle.loads(_f.read())

            orig_frame = np.ndarray(shape=data["shape"], dtype=data["dtype"], buffer=data["frame"])
            frame_no = data["frame_no"]
            fg_mask, frame, contours = process_frame(orig_frame)
            if len(contours):
                logger.info(f"[{frame_no}] Movement Noticed: {len(contours)}")
            else:
                logger.info(f"[{frame_no}] No movements Noticed: {len(contours)}")

            cv.imshow("Frame", frame)
            cv.imshow("FG Mask", fg_mask)
            key = cv.waitKey(0)
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return 0

    source = get_source()
    if source is None:
        logger.info("Unable to capture input source.")
        return 1

    for frame_no, i_frame in enumerate(get_video_stream(source)):
        key = cv.waitKey(1)
        if key == ord("q"):
            source.release()
            break

        fg_mask, frame, contours = process_frame(i_frame)
        # if contours is empty or one of the contours returns valid area
        # then there's movement
        if len(contours) and any(map(is_valid_contour_area, contours)):
            logger.info(f"[{frame_no}] Movement Noticed: {len(contours)}")

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fg_mask)

    cv.destroyAllWindows()
    return 0


def batch_write(output_df, batch_id):
    logger.info(f"inside foreachBatch for {batch_id=}, rows in passed {output_df.count()=}")


@udf(returnType=PROCESSED_FRAME_SCHEMA)
def process_frame_udf(data: bytes):
    recv_data = pickle.loads(data)
    orig_frame = np.ndarray(shape=recv_data["shape"], dtype=recv_data["dtype"], buffer=recv_data["frame"])
    frame_no = recv_data["frame_no"]
    frame = cv.resize(orig_frame, (W, H))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (21, 21), 0)
    bg_sub = cv.createBackgroundSubtractorKNN()
    mask = bg_sub.apply(blur)
    contours = []
    fgmask_contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in fgmask_contours:
        if cv.contourArea(cnt) < CONTOUR_AREA * 4:
            continue

        contours.append(cnt)

    is_movement_present = len(contours) > 0
    j_data = {
        "frame_no": frame_no,
        "frame": pickle.dumps(orig_frame),
        "frame_shape": [*recv_data["shape"]],
        "frame_dtype": recv_data["dtype"].str,
        "contours": pickle.dumps(contours),
        "contours_len": len(contours),
        "contours_dtype": contours[0].dtype.str if len(contours) else "uint8",
        "is_movement_present": is_movement_present,
        "unknown": "",
    }
    return j_data


def handle_socket_stream(sink: SinkType, port=9999, directory="data"):
    curr_dir = Path().absolute()
    python_exe = curr_dir / "venv" / "Scripts" / "python.exe"
    conf = SparkConf()
    conf.set("spark.pyspark.python", python_exe.as_posix())
    spark = (
        SparkSession.builder.appName("SparkStreamApp")
        .config(conf=conf)
        .getOrCreate()
    )
    read_stream = spark.readStream.format("binaryFile" if sink == SinkType.file else sink.name)
    if sink == SinkType.file:
        read_stream = (
            read_stream.schema(SOURCE_FILE_SCHEMA)
            # .option("cleanSource", "delete")
        )
        stream_df = read_stream.load(f"{directory}/")
    elif sink == SinkType.socket:
        read_stream = read_stream.schema(SOURCE_SOCKET_SCHEMA).option("host", "localhost").option("port", 9999)
        stream_df = read_stream.load()

    COL = "value" if sink == SinkType.socket else "content"
    stream_df.isStreaming
    stream_df.printSchema()
    df = stream_df.select(process_frame_udf(COL).alias("processed"))
    df.printSchema()
    df = (
        df.withColumn("frame_no", df.processed.frame_no)
        .withColumn('frame', df.processed.frame)
        .withColumn('frame_shape', df.processed.frame_shape)
        .withColumn('frame_dtype', df.processed.frame_dtype)
        .withColumn('contours', df.processed.contours)
        .withColumn('contours_len', df.processed.contours_len)
        .withColumn('contours_dtype', df.processed.contours_dtype)
        .withColumn("is_movement_present", df.processed.is_movement_present)
        .withColumn("unknown", df.processed.unknown)
        .drop("processed")
    )
    # df = df.groupBy("is_movement_present").agg(F.collect_list("frame_no"))
    query = (
        df.writeStream
        # .foreachBatch(batch_write)
        .outputMode("append")
        .format("console")
        .option("truncate", "true")
        # .option('checkpointLocation', save_loc + "/_checkpoint")
        # .start(save_loc)
        .start()
    )
    query.awaitTermination()


def main():
    # return basic_processing(read_dir=True)
    sock = DataGenStreamThread(
        sink=SinkType.file,
        directory="data",
        source=get_source(),
        daemon=True,
        # port=9999
    )
    sock.start_gen()
    logger.info("Waiting before starting spark streaming...")
    time.sleep(2)
    logger.info("Finished waiting. Starting spark streaming...")
    handle_socket_stream(sink=SinkType.file, directory="data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
