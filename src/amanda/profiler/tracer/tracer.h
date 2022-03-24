// This file describes the api[class and function] of trace process.

// Flush all remaining CUPTI buffers before resetting the device.
// This can also be called in the cudaDeviceReset callback.
void activityFlushAll();

// Init the trace context
void initTrace();
// Finish the trace context, just flush the buffer
void finishTrace();