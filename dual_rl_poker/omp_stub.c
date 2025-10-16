#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int shm_open(const char *name, int oflag, mode_t mode) {
    char path[256];
    if (!name) {
        return -1;
    }
    snprintf(path, sizeof(path), "/tmp/%s", name[0] == '/' ? name + 1 : name);
    fprintf(stderr, "[omp_stub] shm_open %s flags=%d mode=%o\n", path, oflag, mode);
    int flags = oflag;
    if ((oflag & O_CREAT) != 0) {
        flags |= O_RDWR;
    }
    int fd = open(path, flags, mode);
    if (fd == -1 && (oflag & O_CREAT) == 0) {
        fd = open(path, O_CREAT | O_RDWR, mode);
    }
    return fd;
}

int shm_unlink(const char *name) {
    if (!name) {
        return 0;
    }
    char path[256];
    snprintf(path, sizeof(path), "/tmp/%s", name[0] == '/' ? name + 1 : name);
    fprintf(stderr, "[omp_stub] shm_unlink %s\n", path);
    unlink(path);
    return 0;
}

void omp_set_num_threads(int n) {}
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
