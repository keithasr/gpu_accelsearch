/*
 * app_utilities.h
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */

#ifndef APP_UTIL_H_
#define APP_UTIL_H_

#include <stdio.h>

void logMessage(const char *message, ...);
void throwError(const char *errorMessage, ...);

#endif /* APP_UTIL_H_ */
