#include <sys/time.h>
#include <stddef.h>

/* Thread functions */
int omp_get_thread_num(void) { return 0; }
int omp_get_num_threads(void) { return 1; }
int omp_get_max_threads(void) { return 1; }
void omp_set_num_threads(int num) { (void)num; }
int omp_get_num_procs(void) { return 1; }
int omp_in_parallel(void) { return 0; }
void omp_set_dynamic(int dynamic) { (void)dynamic; }
int omp_get_dynamic(void) { return 0; }
void omp_set_nested(int nested) { (void)nested; }
int omp_get_nested(void) { return 0; }
int omp_get_thread_limit(void) { return 1; }
void omp_set_max_active_levels(int levels) { (void)levels; }
int omp_get_max_active_levels(void) { return 1; }
int omp_get_level(void) { return 0; }
int omp_get_ancestor_thread_num(int level) { (void)level; return 0; }
int omp_get_team_size(int level) { (void)level; return 1; }
int omp_get_active_level(void) { return 0; }

/* Timing functions */
double omp_get_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

double omp_get_wtick(void) {
    return 1.0e-6;
}
