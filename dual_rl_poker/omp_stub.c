#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int shm_open(const char *name, int oflag, mode_t mode) {
    return -1;
}

int shm_unlink(const char *name) {
    return 0;
}

void omp_set_num_threads(int n) {}
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
