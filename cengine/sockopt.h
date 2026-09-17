/* Set a connected TCP socket into non-blocking mode. */
#ifndef FB_SOCKOPT_H
#define FB_SOCKOPT_H

#include "util.h"

static inline int fb_set_nonblocking(int fd) {
#if defined(_WIN32)
    u_long mode = 1;
    return ioctlsocket((SOCKET)fd, FIONBIO, &mode) == 0 ? 0 : -1;
#else
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl < 0) return -1;
    return fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0 ? 0 : -1;
#endif
}

#endif
