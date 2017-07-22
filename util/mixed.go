package util

import (
	log "github.com/sirupsen/logrus"
	"strconv"
)

// search string in slice
func StringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

// string slice to float slice cast
func StringToFloat(strrecord []string) []float64 {
	var fltrecord []float64
	if len(strrecord) == 0 {
		log.WithFields(log.Fields{
			"event": "empty_parameter",
			"topic": "util_function",
			"key":   "stringToFloat",
		}).Info("empty slice of string")
	}
	for _, strval := range strrecord {
		if fltval, err := strconv.ParseFloat(strval, 64); err == nil {
			fltrecord = append(fltrecord, fltval)
		}
	}
	log.WithFields(log.Fields{
		"event": "result_info",
		"topic": "stringToFloat",
		"key":   len(fltrecord),
	}).Info("fltrecord length")
	return fltrecord
}
