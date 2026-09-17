/* flybrain-c: small cross-platform utilities (time, files, sockets). */
#ifndef FB_UTIL_H
#define FB_UTIL_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <pthread.h>
  #include <signal.h>
#endif

/* monotonic wall clock in seconds (arbitrary origin) */
double fb_now(void);
/* sleep milliseconds */
void fb_sleep_ms(int ms);

/* read a whole file; returns malloc'd buffer (NUL-terminated for text use),
 * sets *out_len. Returns NULL on failure. */
uint8_t *fb_read_file(const char *path, size_t *out_len);
int fb_remove_file(const char *path); /* 0 on success (or already gone) */

/* one-time socket library init (WSAStartup on Windows); returns 0 on success */
int fb_net_init(void);

/* create a listening TCP socket on 127.0.0.1:port; returns fd or -1 */
int fb_tcp_listen(int port);

/* accept a client (blocking); returns fd or -1 */
int fb_tcp_accept(int listen_fd);

/* read up to cap bytes; returns bytes read (0 = closed), -1 = would block */
int fb_recv(int fd, uint8_t *buf, size_t cap);

/* send all bytes; returns 0 on success, -1 on error */
int fb_send_all(int fd, const uint8_t *buf, size_t len);

/* wait for readability (returns >0 if readable, 0 timeout, <0 error) */
int fb_wait_readable(int fd, int timeout_ms);

/* close socket */
void fb_close(int fd);

#endif
