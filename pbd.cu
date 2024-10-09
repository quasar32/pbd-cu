#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>

#define FPS 60
#define N_BEADS 8 

#define DT (1.0f / FPS)
#define N_STEPS 100
#define SDT (DT / N_STEPS)
#define STS (N_STEPS * FPS)

struct bead {
  float radius;
  float mass;
  float2 pos;
  float2 prev_pos;
  float2 vel;
};

struct wire {
  float2 pos;
  float radius;
};

static bead(*host_groups)[N_BEADS]; 
static bead(*device_groups)[N_BEADS]; 
static wire host_wire = {{0.0f, 0.0f}, 0.8f};
static __constant__ __device__ wire device_wire = {{0.0f, 0.0f}, 0.8f};
static int n_groups = 1;
static int ends_only;
static FILE **csvs;

__constant__ __device__ float2 gravity = {0.0F, -10.0F};

__device__ float2 operator*(float2 a, float b) {
  return (float2) {a.x * b, a.y * b};
}

__device__ float2 operator-(float2 a, float2 b) {
  return (float2) {a.x - b.x, a.y - b.y};
}

__device__ float2 operator+(float2 a, float2 b) {
  return (float2) {a.x + b.x, a.y + b.y};
}

__device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y; 
}

__device__ float length(float2 a) {
  return sqrtf(a.x * a.x + a.y * a.y);
}

__device__ float2 operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
  return a; 
}

__device__ float2 operator/(float2 a, float b) {
  return (float2) {a.x / b, a.y / b};
}

__device__ void start_step(bead *a) {
  a->vel += gravity * SDT; 
  a->prev_pos = a->pos;
  a->pos += a->vel * SDT; 
}

__device__ void end_step(bead *a) {
  a->vel = a->pos - a->prev_pos;
  a->vel = a->vel * STS;
}

__device__ void bead_col(bead *a, bead *b) {
  float2 dir = b->pos - a->pos;
  float d = length(dir);
  if (d == 0.0f || d > a->radius + b->radius)
    return;
  dir = dir / d;
  float corr = (a->radius + b->radius - d) / 2.0f;
  a->pos = a->pos - dir * corr;
  b->pos = b->pos + dir * corr;
  float v0a = dot(a->vel, dir); 
  float v0b = dot(b->vel, dir); 
  float ma = a->mass;
  float mb = b->mass;
  float mt = ma + mb;
  float vc = ma * v0a + mb * v0b;
  float v1a = (vc - mb * (v0a - v0b)) / mt;
  float v1b = (vc - ma * (v0b - v0a)) / mt; 
  a->vel = a->vel + dir * (v1a - v0a); 
  b->vel = b->vel + dir * (v1b - v0b); 
}

__device__ void keep_on_wire(bead *a, wire *b) {
  float2 dir = a->pos - b->pos;
  float len = length(dir); 
  if (len == 0.0f)
    return;
  dir = dir / len;
  float lambda = b->radius - len;
  a->pos = a->pos + dir * lambda;
}

__global__ void update_sim(bead(*groups)[N_BEADS], int n_groups) {
  int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (group_idx >= n_groups)
    return;
  bead *beads = groups[group_idx];
  for (int s = 0; s < N_STEPS; s++) {
    int i, j;
    for (i = 0; i < N_BEADS; i++)
      start_step(beads + i);
    for (i = 0; i < N_BEADS; i++)
      keep_on_wire(beads + i, &device_wire);
    for (i = 0; i < N_BEADS; i++)
      end_step(beads + i);
    for (i = 0; i < N_BEADS; i++) {
      for (j = 0; j < i; j++)
        bead_col(beads + i, beads + j);
    }
  }
}

static void die(const char *fn, int err) {
  fprintf(stderr, "%s(%d)\n", fn, err);
  exit(EXIT_FAILURE);
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "eg:")) != -1) {
    switch (c) {
    case 'e':
      /* output only first and last frame */
      ends_only = 1;
      break;
    case 'g':
      /* number of groups */
      if (sscanf(optarg, "%d\n", &n_groups) != 1) {
        fprintf(stderr, "group number is invalid\n");
        exit(1);
      }
      if (n_groups < 0) {
        fprintf(stderr, "negative groups\n");
        exit(1);
      }
      if (n_groups > 65536) {
        fprintf(stderr, "too many groups\n");
        exit(1);
      }
      break;
    case '?': 
      exit(1);
    }
  }
}

static void init_beads(void) {
  host_groups = new bead[n_groups][N_BEADS]; 
  for (int i = 0; i < n_groups; i++) { 
    float r = 0.1f;
    float rot = 0.0f;
    for (int j = 0; j < N_BEADS; j++) {
      bead *bd = &host_groups[i][j];
      bd->radius = r;
      bd->mass = (float) M_PI * r * r; 
      bd->pos.x = host_wire.pos.x + host_wire.radius * cosf(rot);
      bd->pos.y = host_wire.pos.y + host_wire.radius * sinf(rot);
      rot += (float) M_PI / N_BEADS;
      r = 0.05f + drand48() * 0.1f;
    }
  }
  cudaError_t err = cudaMalloc((void **) &device_groups, 
      n_groups * N_BEADS * sizeof(bead)); 
  if (err != cudaSuccess)
    die("cudaMalloc", err);
  err = cudaMemcpy(device_groups, host_groups, 
      n_groups * N_BEADS * sizeof(bead), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    die("cudaMemcpy", err);
}

#if 0
static void update_sim(void) {
  cl_event ev;
  cl_int err = clEnqueueNDRangeKernel(cmdq, kernel, 1, NULL,
      (size_t[]) {n_groups}, NULL, 0, NULL, &ev);
  if (err != CL_SUCCESS)
    die("clEnqueueTask", err);
  err = clWaitForEvents(1, &ev);
  if (err != CL_SUCCESS)
    die("clWaitForEvents", err);
  cl_ulong start, end;
  err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &start, NULL);
  if (err != CL_SUCCESS)
    die("clGetEventProfilingInfo", err);
  err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &end, NULL);
  if (err != CL_SUCCESS)
    die("clGetEventProfilingInfo", err);
  elapsed += end - start;
  clReleaseEvent(ev);
  err = clEnqueueReadBuffer(cmdq, groups_mem, CL_TRUE, 0, 
      n_groups * sizeof(*host_groups), host_groups, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    die("clEnqueueReadBuffer", err);
}
#endif

static FILE *open_csv(int i) {
  char buf[64]; 
  sprintf(buf, "out%03d.csv", i); 
  sprintf(buf, "out%06d.csv", i); 
  FILE *csv = fopen(buf, "wb");
  if (!csv)
    die("fopen", errno);
  return csv;
}

static void open_all(void) {
  csvs = new FILE*[n_groups];
  for (int i = 0; i < n_groups; i++) {
    char buf[64]; 
    sprintf(buf, "out%03d.csv", i); 
    csvs[i] = open_csv(i); 
  }
}

static void print_header(FILE *csv) {
  fprintf(csv, "f,t,x,y,r\n");
}

static void print_sim_one(FILE *csv, bead *host_groups, int frame) {
  for (int i = 0; i < N_BEADS; i++) {
    fprintf(csv, "%d,%d,%f,%f,%f\n", frame, 0, 
        host_groups[i].pos.x, host_groups[i].pos.y, host_groups[i].radius); 
  }
  fprintf(csv, "%d,%d,%f,%f,%f\n", frame, 1, 
      host_wire.pos.x, host_wire.pos.y, host_wire.radius); 
}

static void print_sim_all(int frame) {
  for (int i = 0; i < n_groups; i++) 
    print_sim_one(csvs[i], host_groups[i], frame);
}

static void update_sim_def(void) {
  int n_blocks = (n_groups + 255) / 256;
  update_sim<<<n_blocks, 256>>>(device_groups, n_groups);
  int err = cudaGetLastError();
  if (err != cudaSuccess) 
    die("update_sim", err);
}

static void copy_device_beads(bead(*host_groups)[N_BEADS]) {
    int err = cudaMemcpy(host_groups, device_groups, 
        n_groups * N_BEADS * sizeof(bead), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) 
      die("cudaMemcpy", err);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  init_beads();
  if (ends_only) {
    int f;
    for (f = 0; f < 10 * FPS; f++) 
      update_sim_def();
    bead(*end)[N_BEADS] = new bead[n_groups][N_BEADS];
    copy_device_beads(end);
    for (int i = 0; i < n_groups; i++) {
      FILE *csv = open_csv(i);
      print_header(csv);
      print_sim_one(csv, host_groups[i], 0);
      print_sim_one(csv, end[i], 1);
      fclose(csv);
    }
    delete[] end;
  } else {
    open_all();
    for (int i = 0; i < n_groups; i++)
      print_header(csvs[i]);
    int f;
    for (f = 0; f < 10 * FPS; f++) {
      print_sim_all(f);
      update_sim_def();
      copy_device_beads(host_groups);
    }
    print_sim_all(f);
    for (int i = 0; i < n_groups; i++) 
      fclose(csvs[i]);
    delete[] csvs;
  }
  delete[] host_groups;
  cudaFree(device_groups);
  return 0;
}
