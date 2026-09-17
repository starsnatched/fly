/* SHA-1 (RFC 3174) + base64, for the WebSocket opening handshake. */
#ifndef FB_SHA1_H
#define FB_SHA1_H

#include <stdint.h>
#include <stddef.h>

void fb_sha1(const uint8_t *data, size_t len, uint8_t out[20]);
void fb_base64_encode(const uint8_t *data, size_t len, char *out /* len*4/3+4 */);

#endif
