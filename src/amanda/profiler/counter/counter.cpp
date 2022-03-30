#include "counter.h"

counterControler* counter::getControler() {
	return &this->controler;
}

counter::counter() {
	this->filePath = "metrics_record.txt";
	this->kindFlag = 0;
	this->countMode = Counter::OFFLINE_AND_ONLINE;
	
	this->setMetrics(0);
}

counter::counter(std::string _filePath):filePath(_filePath) {
	this->kindFlag = 0;
	this->countMode = Counter::OFFLINE_AND_ONLINE;

	this->setMetrics(0);
}

counter::counter(unsigned long _kindFlag):kindFlag(_kindFlag) {
	this->filePath = "metrics_record.txt";
	this->countMode = Counter::OFFLINE_AND_ONLINE;

	this->setMetrics(_kindFlag);
}

counter::counter(unsigned long _kindFlag, std::string _filePath):kindFlag(_kindFlag), filePath(_filePath) {
	this->countMode = Counter::OFFLINE_AND_ONLINE;

	this->setMetrics(_kindFlag);
}

counter::~counter() {}

void counter::setFilePath(std::string _filePath) {
	this->filePath = _filePath; 
}

void counter::setKindFlag(unsigned long _kindFlag) {
	this->kindFlag = _kindFlag;
	
	this->setMetrics(_kindFlag);
}

std::string counter::getFilePath() {
	return this->filePath;
}

unsigned long counter::getKindFlag() {
	return this->kindFlag;
}

void counter::onlineAnalysisOnly() {
	this->countMode = Counter::ONLINE_ONLY;
}

void counter::offlineAnalysisOnly() {
	this->countMode = Counter::OFFLINE_ONLY;
}

void counter::setCountDevice(int _deviceNum) {
	this->controler.deviceNum = _deviceNum;
}

// Flush all data now in controler 
void counter::setCountParams(int _deviceNum, std::vector<std::string> _metricNames) {
	counterControler new_controler;
	new_controler.deviceNum = _deviceNum;
	new_controler.metricNames = _metricNames;
	this->controler = new_controler;
}

void counter::clearData() {
	this->countData.clear();
}