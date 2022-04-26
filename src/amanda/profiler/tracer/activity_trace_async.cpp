/*
 * Copyright 2011-2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 * using asynchronous handling of activity buffers.
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <pthread.h>
#include <fstream>

#include "tracer.h"
#include "kind_enable.h"

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;
static uint64_t endTimestamp;
static std::fstream traceFile;
static pthread_mutex_t traceCount_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_rwlock_t rwlock_file;
static pthread_rwlock_t rwlock_traceData;
static const char* api_name;

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

  #define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

// Buffer that stored the trace data
// #define BUF_SIZE (sizeof(CUpti_ActivityAPI))
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

/**
 * Print Activity
 * 
 */

static void
printActivity(CUpti_Activity *record)
{
  int rc;
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
      printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
      printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
             context->contextId, context->deviceId,
             getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
             (int) context->nullStreamId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *) record;
      printf("MEMCPY %s [ %llu - %llu, %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
             (unsigned long long) memcpy->start,
             (unsigned long long) memcpy->end,
             (unsigned long long) (memcpy->end - memcpy->start),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *) record;
      printf("MEMSET value=%u [ %llu - %llu, %llu ] device %u, context %u, stream %u, correlation %u\n",
             memset->value,
             (unsigned long long) memset->start,
             (unsigned long long) memset->end,
             (unsigned long long) (memset->end - memset->start),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *) record;
      printf("%s \"%s\" [ %llu - %llu, %llu ] device %u, context %u, stream %u, correlation %u",
             kindString,
             kernel->name,
             (unsigned long long) kernel->start,
             (unsigned long long) kernel->end,
             (unsigned long long) (kernel->end - kernel->start),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);

      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu, %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) api->start,
             (unsigned long long) api->end,
             (unsigned long long) (api->end - api->start),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu, %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) api->start,
             (unsigned long long) api->end,
             (unsigned long long) (api->end - api->start),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        printf("NAME  %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      case CUPTI_ACTIVITY_OBJECT_STREAM:
        printf("NAME %s %u %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      default:
        printf("NAME %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               name->name);
        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER id %u [ %llu ], name %s, domain %s\n",
             marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA:
    {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
             marker->id, marker->color, marker->category,
             (unsigned long long) marker->payload.metricValueUint64,
             marker->payload.metricValueDouble);
      break;
    }
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
      printf("OVERHEAD %s [ %llu, %llu, %llu ] %s id %u\n",
             getActivityOverheadKindString(overhead->overheadKind),
             (unsigned long long) overhead->start,
             (unsigned long long) overhead->end,
             (unsigned long long) (overhead->end - overhead->start),
             getActivityObjectKindString(overhead->objectKind),
             getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
      break;
    }
  default:
    printf("  <unknown>\n");
    break;
  }
}

/**
 * Write File.
 *  
 */

static void
writeFileActivity(CUpti_Activity *record)
{
  int rc;
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "DEVICE " << device->name << " " << device->id << ", "
                << "capability " << device->computeCapabilityMajor << "." << device->computeCapabilityMinor << ", "
                << "global memory " << "(bandwidth " << (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024) << " GB/s, " << "size " << (unsigned int) (device->globalMemorySize / 1024 / 1024) << " MB), "
                << "multiprocessors " << device->numMultiprocessors << ", "
                << "clock " << (unsigned int) (device->coreClockRate / 1000) << " MHz" << std::endl; 
      rc = pthread_rwlock_unlock(&rwlock_file);
      
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "DEVICE_ATTRIBUTE " << attribute->attribute.cupti << ", "
                << "device " << attribute->deviceId << ", "
                << "value=" << (unsigned long long)(attribute->value.vUint64) << std::endl; 
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "CONTEXT " << context->contextId << ", "
                << "device " << context->deviceId << ", "
                << "compute API " << getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind) << ", "
                << "NULL stream " << (int)(context->nullStreamId) << std::endl;  
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "MEMCPY " << getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind) << " "
              << "[" << (unsigned long long) memcpy->start << " - " << (unsigned long long) memcpy->end << ", " << (unsigned long long) (memcpy->end - memcpy->start) << "] "
              << "device: " << memcpy->deviceId << " "
              << "context: " << memcpy->contextId << " "
              << "stream: " << memcpy->streamId << " "
              << "correlation: " << memcpy->correlationId << "/" << memcpy->runtimeCorrelationId << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile <<  "MEMSET value=" << memset->value << " "
              << "[" << (unsigned long long) memset->start << " - " << (unsigned long long) memset->end << ", " << (unsigned long long) (memset->end - memset->start) << "] "
              << "device: " << memset->deviceId << " "
              << "context: " << memset->contextId << " "
              << "stream: " << memset->streamId << " "
              << "correlation: " << memset->correlationId << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *) record;
      
      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << kindString << " " << kernel->name << " "
              << "[" << (unsigned long long)kernel->start << " - " << (unsigned long long)kernel->end << ", " << (unsigned long long)(kernel->end - kernel->start) << "] "
              << "device: " << kernel->deviceId << " "
              << "context: " << kernel->contextId << " "
              << "stream: " << kernel->streamId << " "
              << "correlation: " << kernel->correlationId;

      traceFile << "    grid [" << kernel->gridX << "," << kernel->gridY << "," << kernel->gridZ << "], "
                << "shared memory (static " << kernel->staticSharedMemory << ", dynamic " << kernel->dynamicSharedMemory << ")" << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      
      rc = pthread_rwlock_wrlock(&rwlock_file);
      cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api->cbid, &api_name);
      traceFile << "DRIVER cbid=" <<  api->cbid << " " << "api_name=" << api_name << " "
              << "[" << (unsigned long long) api->start << " - " << (unsigned long long) api->end << ", " << (unsigned long long) (api->end - api->start) << "] "
              << "process: " << api->processId << " "
              << "thread: " << api->threadId << " "
              << "correlation: " << api->correlationId << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &api_name);
      traceFile << "RUNTIME cbid=" <<  api->cbid << " " << "api_name=" << api_name << " "
              << "[" << (unsigned long long) api->start << " - " << (unsigned long long) api->end << ", " << (unsigned long long) (api->end - api->start) << "] "
              << "process: " << api->processId << " "
              << "thread: " << api->threadId << " "
              << "correlation: " << api->correlationId << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:

        rc = pthread_rwlock_wrlock(&rwlock_file);
        traceFile << "NAME" << "  "
                  << getActivityObjectKindString(name->objectKind) << " "
                  << getActivityObjectKindId(name->objectKind, &name->objectId) << " "
                  << getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE) << " "
                  << getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId) << ", "
                  << "name " << name->name << std::endl;  
        rc = pthread_rwlock_unlock(&rwlock_file);

        break;
      case CUPTI_ACTIVITY_OBJECT_STREAM:

        rc = pthread_rwlock_wrlock(&rwlock_file);
        traceFile << "NAME" << "  "
                  << getActivityObjectKindString(name->objectKind) << " " 
                  << getActivityObjectKindId(name->objectKind, &name->objectId) << " "
                  << getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT) << " "
                  << getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId) << " "
                  << getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE) << " "
                  << getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId) << ", "
                  << "name " << name->name << std::endl; 
        rc = pthread_rwlock_unlock(&rwlock_file);

        break;
      default:
        rc = pthread_rwlock_wrlock(&rwlock_file);
        traceFile << "NAME  " << getActivityObjectKindString(name->objectKind) << " "
                  << "id " << getActivityObjectKindId(name->objectKind, &name->objectId) << ", "
                  << "name " << name->name << std::endl; 
        rc = pthread_rwlock_unlock(&rwlock_file);

        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "MARKER id " << marker->id << " " 
              << " [ " << (unsigned long long) marker->timestamp << "], "
              << "name " << marker->name << ", "
              << "domain " << marker->domain << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;

      rc = pthread_rwlock_wrlock(&rwlock_file);
      traceFile << "OVERHEAD " << getActivityOverheadKindString(overhead->overheadKind) << " " 
              << "[" << (unsigned long long) overhead->start << " - " << (unsigned long long) overhead->end << ", " << (unsigned long long) (overhead->end - overhead->start) << "] "
              << getActivityObjectKindString(overhead->objectKind) << " "
              << "id " << getActivityObjectKindId(overhead->objectKind, &overhead->objectId) << std::endl;
      rc = pthread_rwlock_unlock(&rwlock_file);

      break;
    }
  default:
    rc = pthread_rwlock_wrlock(&rwlock_file);
    traceFile << "  <unknown>" << std::endl;
    rc = pthread_rwlock_unlock(&rwlock_file);
    break;
  }
}

/**
 * Store Value.
 *  
 */

static void
storeValueActivity(CUpti_Activity *record)
{
  int rc;
  unsigned short dataTypeFlag = globalTracer_pointer->getDataTypeFlag();
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *) record;
      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x1) ) {
        Tracer::traceData_rt traceData;
        traceData.startTime = (unsigned long long) memcpy->start;
        traceData.endTime = (unsigned long long) memcpy->end;
        traceData.durationTime = (unsigned long long) (memcpy->end - memcpy->start);
      
        traceData.deviceId = memcpy->deviceId;
        traceData.contextId = memcpy->contextId;
        traceData.streamId = memcpy->streamId;
        traceData.correlationId = memcpy->correlationId;

        traceData.kind = "MEMCPY";
        traceData.name = getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind);

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        globalTracer_pointer->traceData_rt.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }
      
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *) record;

      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x1) ) {
        Tracer::traceData_rt traceData;
        traceData.startTime = (unsigned long long) memset->start;
        traceData.endTime = (unsigned long long) memset->end;
        traceData.durationTime = (unsigned long long) (memset->end - memset->start);
      
        traceData.deviceId = memset->deviceId;
        traceData.contextId = memset->contextId;
        traceData.streamId = memset->streamId;
        traceData.correlationId = memset->correlationId;

        traceData.kind = "MEMSET";

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        globalTracer_pointer->traceData_rt.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }

      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *) record;
      
      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x1) ) {
        Tracer::traceData_rt traceData;
        traceData.startTime = (unsigned long long) kernel->start;
        traceData.endTime = (unsigned long long) kernel->end;
        traceData.durationTime = (unsigned long long) (kernel->end - kernel->start);
      
        traceData.deviceId = kernel->deviceId;
        traceData.contextId = kernel->contextId;
        traceData.streamId = kernel->streamId;
        traceData.correlationId = kernel->correlationId;

        traceData.kind = kindString;
        traceData.name = kernel->name;

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        globalTracer_pointer->traceData_rt.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }

      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;

      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x2) ) {
        Tracer::traceData_api traceData;
        traceData.startTime = (unsigned long long) api->start;
        traceData.endTime = (unsigned long long) api->end;
        traceData.durationTime = (unsigned long long) (api->end - api->start);
      
        traceData.processId = api->processId;
        traceData.threadId = api->threadId;
        traceData.correlationId = api->correlationId;

        traceData.kind = "DRIVER";

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api->cbid, &api_name);
        traceData.name = api_name;
        globalTracer_pointer->traceData_api.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }

      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;

      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x2) ) {
        Tracer::traceData_api traceData;
        traceData.startTime = (unsigned long long) api->start;
        traceData.endTime = (unsigned long long) api->end;
        traceData.durationTime = (unsigned long long) (api->end - api->start);
      
        traceData.processId = api->processId;
        traceData.threadId = api->threadId;
        traceData.correlationId = api->correlationId;

        traceData.kind = "RUNTIME";

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &api_name);
        traceData.name = api_name;
        globalTracer_pointer->traceData_api.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }

      break;
    } 
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;

      if ( (dataTypeFlag == 0) || (dataTypeFlag & 0x4) ) {
        Tracer::traceData_oh traceData;
        traceData.startTime = (unsigned long long) overhead->start;
        traceData.endTime = (unsigned long long) overhead->end;
        traceData.durationTime = (unsigned long long) (overhead->end - overhead->start);

        traceData.kind = "OVERHEAD";
        traceData.overheadKind = getActivityOverheadKindString(overhead->overheadKind);
        traceData.objectKind = getActivityObjectKindString(overhead->objectKind);
        traceData.objectId = getActivityObjectKindId(overhead->objectKind, &overhead->objectId);

        rc = pthread_rwlock_wrlock(&rwlock_traceData);
        globalTracer_pointer->traceData_oh.push_back(traceData);
        rc = pthread_rwlock_unlock(&rwlock_traceData);
      }

      break;
    }
  default:
    break;
  }
}

/**
 * Subscriber Function.
 * 
 */

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
        if (globalTracer_pointer->getTraceMode() == Tracer::OFFLINE_AND_ONLINE | globalTracer_pointer->getTraceMode() == Tracer::OFFLINE_ONLY) {
          writeFileActivity(record);
        }
        if (globalTracer_pointer->getTraceMode() == Tracer::OFFLINE_AND_ONLINE | globalTracer_pointer->getTraceMode() == Tracer::ONLINE_ONLY) {
          storeValueActivity(record);
        }
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}


/**
 * User API.
 * 
 */


void tracer::activityFlushAll()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));
}

void tracer::initTrace()
{
  size_t attrValue = 0, attrValueSize = sizeof(size_t);

  // Add the traceCount, if traceCount == 1, open the file.
  int rc = pthread_mutex_lock(&traceCount_mutex);
  this->traceCount++;
  if (this->traceCount == 1) {
    traceFile.open(this->filePath, std::ios::app);
  }
  rc = pthread_mutex_unlock(&traceCount_mutex);

  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  enableActivityKind(this->kindFlag);

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_SIZE", (long long unsigned)attrValue);

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_PROFILING_SEMAPHORE_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
  traceFile << "Update startTimeStamp: " << startTimestamp << " thread: " << pthread_self() << std::endl;
  std::cout << "Update startTimeStamp: " << startTimestamp << " thread: " << pthread_self() << std::endl;
  if (this->traceMode == Tracer::OFFLINE_AND_ONLINE || this->traceMode == Tracer::ONLINE_ONLY){
    this->startTimeLists.push_back((unsigned long long)startTimestamp);
  }  
}

void tracer::finishTrace()
{   
   CUPTI_CALL(cuptiActivityFlushAll(0));
   disableActivityKind(this->kindFlag);

   CUPTI_CALL(cuptiGetTimestamp(&endTimestamp));
   if (this->traceMode == Tracer::OFFLINE_AND_ONLINE || this->traceMode == Tracer::ONLINE_ONLY){
    this->endTimeLists.push_back((unsigned long long)endTimestamp);
   }
   traceFile << "Update endTimeStamp: " << endTimestamp << " thread: " << pthread_self() << std::endl;
   std::cout << "Update endTimeStamp: " << endTimestamp << " thread: " << pthread_self() << std::endl;

   int rc = pthread_mutex_lock(&traceCount_mutex);
   this->traceCount--;
   if (this->traceCount == 0) {
     traceFile.close();
   }
   pthread_mutex_unlock(&traceCount_mutex);
}
