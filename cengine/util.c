/* Cross-platform shims: time, files, sockets. */
#include "util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
static LARGE_INTEGER qpc_freq;
static int qpc_init = 0;

double fb_now(void) {
    if (!qpc_init) { QueryPerformanceFrequency(&qpc_freq); qpc_init = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)qpc_freq.QuadPart;
}

void fb_sleep_ms(int ms) { Sleep((DWORD)ms); }

int fb_net_init(void) {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
}

#else
double fb_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void fb_sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

int fb_net_init(void) {
    signal(SIGPIPE, SIG_IGN);
    return 0;
}
#endif

uint8_t *fb_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    buf[sz] = 0;
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

int fb_remove_file(const char *path) {
    if (remove(path) != 0 && errno != ENOENT) return -1;
    return 0;
}

int fb_copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }
    char buf[65536];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), in)) > 0)
        fwrite(buf, 1, r, out);
    int err = ferror(in) || ferror(out) || fclose(out) != 0;
    fclose(in);
    return err ? -1 : 0;
}

int fb_tcp_listen(int port) {
    int fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { fb_close(fd); return -1; }
    if (listen(fd, 16) != 0) { fb_close(fd); return -1; }
    return fd;
}

int fb_tcp_accept(int listen_fd) {
    struct sockaddr_in peer;
#if defined(_WIN32)
    int plen = (int)sizeof(peer);
#else
    socklen_t plen = sizeof(peer);
#endif
    return (int)accept(listen_fd, (struct sockaddr *)&peer, &plen);
}

int fb_recv(int fd, uint8_t *buf, size_t cap) {
#if defined(_WIN32)
    int r = (int)recv(fd, (char *)buf, (int)cap, 0);
#else
    ssize_t r = recv(fd, buf, cap, 0);
#endif
    if (r < 0) {
#if defined(_WIN32)
        int err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) return -1;
#else
        if (errno == EAGAIN || errno == EWOULDBLOCK) return -1;
#endif
        return 0; /* treat other errors as closed */
    }
    return (int)r;
}

int fb_send_all(int fd, const uint8_t *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
#if defined(_WIN32)
        int n = (int)send(fd, (const char *)(buf + off), (int)(len - off), 0);
#else
        ssize_t n = send(fd, buf + off, len - off, 0);
#endif
        if (n > 0) { off += (size_t)n; continue; }
        /* would-block on a non-blocking socket: report progress so callers
         * can drop the whole frame instead of writing a partial one (a
         * truncated WS frame desynchronizes the client's parser) */
        return off > 0 ? (int)off : -1;
    }
    return (int)off;
}

int fb_wait_readable(int fd, int timeout_ms) {
    fd_set rfds;
    struct timeval tv;
    FD_ZERO(&rfds);
    FD_SET((unsigned int)fd, &rfds);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int r = select((int)fd + 1, &rfds, NULL, NULL, &tv);
    return r;
}

void fb_close(int fd) {
    if (fd < 0) return;
#if defined(_WIN32)
    closesocket((SOCKET)fd);
#else
    close(fd);
#endif
}
