import logging
import pickle
import socket
import struct  # noqa
import sys
import time
import typing as t
from functools import lru_cache
from threading import Thread

import cv2 as cv
import cv2.typing as ct
import pyspark.sql.functions as f
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

H, W = 480, 640
CONTOUR_AREA = 500
BG_SUBSTRACTOR = cv.createBackgroundSubtractorKNN()


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
    if cv.contourArea(contour) < CONTOUR_AREA * 4:
        return False

    return True


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
        key = cv.waitKey(1)
        if not ret or ord("q") == key or frame is None:
            source.release()
            break

        yield frame


def process_frame(frame: ct.MatLike) -> t.Tuple[ct.MatLike, ct.MatLike, t.Sequence[ct.MatLike]]:
    """
    params: frame: the original frame where the rectangle will be draw
    """
    blur = resize_and_blur(frame)
    fg_mask = get_fg_mask(blur)
    contours = list(get_contours(fg_mask))
    for contour in contours:
        draw_rect(frame, contour)

    return frame, fg_mask, contours


class SocketThread(Thread):
    def __init__(
        self,
        source: t.Optional[cv.VideoCapture] = None,
        port: t.Optional[int] = None,
        daemon: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.port = port or 9999
        self.source = source or get_source()
        if self.source is None:
            raise Exception("Failed to create source")
        super().__init__(*args, **kwargs, daemon=daemon)

    def send_frame(self, frame: ct.MatLike, client: socket.socket) -> bool:
        try:
            pickled_frame = pickle.dumps(frame)
            # packed = struct.pack("Q", pickled_frame)
            client.sendall(pickled_frame)
            return False
        except:  # noqa
            return True

    def start_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("localhost", self.port))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        logger.info("Socket Listening on port %s", str(self.port))
        self.start()
        return self.port

    def run(self) -> None:
        """Overrides Thread.run

        Creates a socket and waits(blocking) for connections
        When a connection is closed, goes back into waiting.
        """
        while True:
            logger.info("Starting socket thread, going to accept")
            (client, addr) = self.sock.accept()
            logger.info("Client Connected %s", addr)
            for frame_no, frame in enumerate(get_video_stream(self.source)):
                logger.info(f"Processing {frame_no}")
                rv = self.send_frame(frame, client)
                if rv:
                    break

            logger.info("Socket Exiting Client Loop")
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                client.close()

    def start(self):
        """Starts the socket thread"""
        Thread.start(self)


def main2(args: t.Optional[t.Sequence[str]] = None):
    source = get_source()
    if source is None:
        logger.info("Unable to capture input source.")
        return 1

    for frame_no, i_frame in enumerate(get_video_stream(source)):
        fg_mask, frame, contours = process_frame(i_frame)
        # if contours is empty or one of the contours returns valid area
        # then there's movement
        if not all([len(contours), any(map(is_valid_contour_area, contours))]):
            logger.info(f"[{frame_no}] No Movement: ", contours)

        cv.imshow("Frame", frame)
        cv.imshow("FG Mask", fg_mask)

    cv.destroyAllWindows()
    return 0


def sample_spark_job(spark: SparkSession):
    # Unix: nc -lk 9999 | Windows: nc -L -p 9999
    # Split the lines into words
    lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
    words = lines.select(f.explode(f.split(lines.value, " ")).alias("word"))
    # Generate running word count
    wordCounts = words.groupBy("word").count()
    query = wordCounts.writeStream.outputMode("complete").format("console").start()
    query.awaitTermination()
    return 0


def batch_write(output_df, batch_id):
    logger.info(f"inside foreachBatch for {batch_id=}, rows in passed {output_df.count()=}")


def handle_socket_stream():
    spark = SparkSession.builder.appName("SparkStreamApp").getOrCreate()
    # return sample_spark_job(spark)

    socket_df = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
    socket_df.isStreaming
    socket_df.printSchema()
    query = (
        socket_df.writeStream
        # .trigger(once=True)
        .foreachBatch(batch_write)
        .format("console")
        #  .option('checkpointLocation', save_loc + "/_checkpoint")
        # .start(save_loc)
        .start()
    )
    # query = query.writeStream.outputMode("complete").format("console").start()
    query.awaitTermination()


def main():
    sock = SocketThread()
    sock.start_socket()
    logger.info("Waiting before starting spark streaming...")
    time.sleep(2)
    logger.info("Finished waiting. Starting spark streaming...")
    handle_socket_stream()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
