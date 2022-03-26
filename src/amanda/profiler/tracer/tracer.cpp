#include "tracer.h"

tracer *globalTracer_pointer = NULL;
pthread_mutex_t tracer_mutex = PTHREAD_MUTEX_INITIALIZER;

tracer::tracer() {
	this->kindFlag = 0;
	this->filePath = "activity_record.txt";
	this->traceMode = Tracer::OFFLINE_AND_ONLINE;

	pthread_mutex_lock(&tracer_mutex);
	if (globalTracer_pointer != nullptr) {
		throw std::runtime_error("There exists a tracer, and it is not allowed to creat another one.");
	}
	globalTracer_pointer = this; 
	pthread_mutex_unlock(&tracer_mutex);
}

tracer::tracer(unsigned long _kindFlag):kindFlag(_kindFlag) {
	this->filePath = "activity_record.txt";
	this->traceMode = Tracer::OFFLINE_AND_ONLINE;

	pthread_mutex_lock(&tracer_mutex);
	if (globalTracer_pointer != nullptr) {
		throw std::runtime_error("There exists a tracer, and it is not allowed to creat another one.");
	}
	globalTracer_pointer = this; 
	pthread_mutex_unlock(&tracer_mutex);
}

tracer::tracer(std::string _filePath):filePath(_filePath) {
	this->kindFlag = 0;
	this->traceMode = Tracer::OFFLINE_AND_ONLINE;

	pthread_mutex_lock(&tracer_mutex);
	if (globalTracer_pointer != nullptr) {
		throw std::runtime_error("There exists a tracer, and it is not allowed to creat another one.");
	}
	globalTracer_pointer = this; 
	pthread_mutex_unlock(&tracer_mutex);
}

tracer::tracer(unsigned long _kindFlag, std::string _filePath):kindFlag(_kindFlag), filePath(_filePath) {
	this->traceMode = Tracer::OFFLINE_AND_ONLINE;

	pthread_mutex_lock(&tracer_mutex);
	if (globalTracer_pointer != nullptr) {
		throw std::runtime_error("There exists a tracer, and it is not allowed to creat another one.");
	}
	globalTracer_pointer = this; 
	pthread_mutex_unlock(&tracer_mutex);
}

tracer::~tracer() {
	pthread_mutex_lock(&tracer_mutex);
	globalTracer_pointer = nullptr; 
	pthread_mutex_unlock(&tracer_mutex);
}

void tracer::setKindFlag(unsigned long _kindFlag) {
	this->kindFlag = kindFlag;
}

void tracer::setFilePath(std::string _filePath) {
	this->filePath = _filePath;
}

void tracer::onlineAnalysisOnly() {
	this->traceMode = Tracer::ONLINE_ONLY;
}

void tracer::offlineAnalysisOnly() {
	this->traceMode = Tracer::OFFLINE_ONLY;
}

unsigned long tracer::getKindFlag() {
	return this->kindFlag;
}

std::string tracer::getFilePath() {
	return this->filePath;
}