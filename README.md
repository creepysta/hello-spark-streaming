# A small project to understand spark streaming

### Goal:

Have a motion detection pipeline from a raw stream of video frames using pyspark and opencv

### Current Condition -

1. A background thread reads frames from the camera
2. The frame is serialized and sent over socket (loopback address on the same device)
3. Spark reads from loopback address

### TODO:

1. De-Serialize the frames on spark streaming df
2. Process the frames to find contours and valid area threshold to detech movement
3. (May be?) Consider using Ny51 cameras for live video stream instead of device camera
