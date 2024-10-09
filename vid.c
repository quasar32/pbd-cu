#include <SDL2/SDL.h>
#include <glad/gl.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <time.h>

#define N_VERTS 256
#define MAX_LOG 256

static GLuint vao;
static GLuint vbo;
static GLuint prog;
static GLint trans_loc;
static GLint scale_loc;

#define VERTS_SIZE (N_VERTS * sizeof(vec2))

typedef float f32;
typedef int i32;

typedef struct vec2 {
  f32 x;
  f32 y;
} vec2;

void die(const char *fmt, ...) {
  va_list ap;  
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(1);
}

static void init_verts(void) {
  f32 theta = 0.0f;
  vec2 *verts = malloc(VERTS_SIZE);
  if (!verts)
    die("malloc: out of memory\n");
  for (i32 i = 0; i < N_VERTS; i++) {
    verts[i].x = cosf(theta);
    verts[i].y = sinf(theta);
    theta += 2.0F * M_PI / (N_VERTS - 1); 
  }
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glCreateBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, VERTS_SIZE, verts, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(struct vec2), NULL);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);
  free(verts);
  verts = NULL;
}

static const char vs_src[] = 
  "#version 330 core\n"
  "layout(location = 0) in vec2 pos;"
  "uniform vec2 trans;"
  "uniform vec2 scale;"
  "void main() {"
    "gl_Position = vec4(trans + pos * scale, 0.0F, 1.0F);"
  "}";

static const char fs_src[] = 
  "#version 330 core\n"
  "out vec4 color;"
  "void main() {"
    "color = vec4(1.0F, 1.0F, 1.0F, 1.0F);"
  "}";

#define N_GLSL 2

struct shader_desc {
  GLenum type;
  const char *src;
};

static struct shader_desc descs[N_GLSL] = {
  {GL_VERTEX_SHADER, vs_src},
  {GL_FRAGMENT_SHADER, fs_src},
};

static void init_prog(void) {
  GLuint shaders[N_GLSL];
  char log[MAX_LOG];
  int success, i;
  for (i = 0; i < N_GLSL; i++) {
    shaders[i] = glCreateShader(descs[i].type);
    glShaderSource(shaders[i], 1, &descs[i].src, NULL);
    glCompileShader(shaders[i]);
    glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shaders[i], MAX_LOG, NULL, log);
      die("shader: %s\n", log);
    }
  }
  prog = glCreateProgram();
  for (i = 0; i < N_GLSL; i++)
    glAttachShader(prog, shaders[i]);
  glLinkProgram(prog);
  for (i = 0; i < N_GLSL; i++) {
    glDetachShader(prog, shaders[i]);
    glDeleteShader(shaders[i]);
  }
  glGetProgramiv(prog, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(prog, MAX_LOG, NULL, log);
    die("program: %s\n", log);
  }
  trans_loc = glGetUniformLocation(prog, "trans");
  scale_loc = glGetUniformLocation(prog, "scale");
}

void init_draw(void) {
  init_verts();
  init_prog();
}

int draw(int frame) {
  static int valid; 
  static int f, t;
  static float x, y, r;
  glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
  glClear(GL_COLOR_BUFFER_BIT);
  glUseProgram(prog);
  glBindVertexArray(vao);
  if (!valid && scanf("%d,%d,%f,%f,%f", &f, &t, &x, &y, &r) != 5)
      return -1;
  valid = 1;
  int n = 0;
  while (f == frame) {
    glUniform2f(trans_loc, x, y);
    glUniform2f(scale_loc, r, r); 
    glDrawArrays(t ? GL_LINE_LOOP : GL_TRIANGLE_FAN, 0, N_VERTS);
    if (scanf("%d,%d,%f,%f,%f", &f, &t, &x, &y, &r) != 5)
      return n ? 0 : -1;
    n++;
  }
  return 1;
}

#define WIDTH 480 
#define HEIGHT 480
#define FPS 60 

static uint8_t pixels[WIDTH * HEIGHT * 3];
static const char *path = "pbd.mp4";

static void encode(AVCodecContext *cctx, AVFrame *frame,
         AVPacket *pkt, AVFormatContext *fmtctx, AVStream *vid) {
  int ret;

  ret = avcodec_send_frame(cctx, frame);
  if (ret < 0)
    die("avcodec_send_frame: %s\n", av_err2str(ret));
  while (ret >= 0) {
    ret = avcodec_receive_packet(cctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) 
      return;
    if (ret < 0)
      die("avcodec_receive_packet: %s\n", ret);
    av_packet_rescale_ts(pkt, cctx->time_base, vid->time_base);
    av_write_frame(fmtctx, pkt);
    av_packet_unref(pkt);
  }
}  

static void render(void) {
  struct SwsContext *sws;
  const uint8_t *src;
  int src_stride;
  const AVOutputFormat *outfmt;
  AVFormatContext *fmtctx;
  int ret;
  const AVCodec *codec;
  AVStream *vid;
  AVCodecContext *cctx;
  AVPacket *pkt;
  AVFrame *frame;
  int i, more;

  sws = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_RGB24,
    WIDTH, HEIGHT, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR,
    NULL, NULL, NULL);
  if (!sws)
    die("sws_getContext\n");
  src = pixels + WIDTH * 3 * (HEIGHT - 1);
  src_stride = -WIDTH * 3;
  outfmt = av_guess_format(NULL, path, NULL);
  if (!outfmt)
    die("av_guess_format\n");
  ret = avformat_alloc_output_context2(&fmtctx, outfmt, NULL, path);
  if (ret < 0)
    die("avformat_free_context: %s\n", av_err2str(ret));
  codec = avcodec_find_encoder(outfmt->video_codec);
  if (!codec)
    die("avcodec_find_encoder\n");
  vid = avformat_new_stream(fmtctx, codec);
  if (!vid)
    die("avformat_new_stream\n");
  cctx = avcodec_alloc_context3(codec);
  if (!cctx)
    die("avcodec_alloc_context3\n");
  vid->codecpar->codec_id = outfmt->video_codec;
  vid->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  vid->codecpar->width = WIDTH;
  vid->codecpar->height = HEIGHT;
  vid->codecpar->format = AV_PIX_FMT_YUV420P;
  vid->codecpar->bit_rate = 400000;
        av_opt_set(cctx, "preset", "ultrafast", 0);
  avcodec_parameters_to_context(cctx, vid->codecpar);
  cctx->time_base.num = 1; 
  cctx->time_base.den = FPS; 
  cctx->framerate.num = FPS; 
  cctx->framerate.den = 1; 
  cctx->gop_size = FPS * 10;
  cctx->max_b_frames = 1;
  avcodec_parameters_from_context(vid->codecpar, cctx);
  ret = avcodec_open2(cctx, codec, NULL);
  if (ret < 0)
    die("avcodec_open2: %s\n", av_err2str(ret));
  ret = avio_open(&fmtctx->pb, path, AVIO_FLAG_WRITE);
  if (ret < 0)
    die("avio_open: %s\n", av_err2str(ret));
      ret = avformat_write_header(fmtctx, NULL);
  if (ret < 0)
    die("avformat_write_header: %s\n", av_err2str(ret));
  pkt = av_packet_alloc();
  if (!pkt)
    die("av_packet_alloc\n");
  frame = av_frame_alloc();
  if (!frame)
    die("av_frame_alloc\n");
  frame->format = cctx->pix_fmt;  
  frame->width = WIDTH;
  frame->height = HEIGHT;
  ret = av_frame_get_buffer(frame, 0);
  if (ret < 0)
    die("av_frame_get_buffer: %s\n", av_err2str(ret));
  i = 0;
  do {
    ret = av_frame_make_writable(frame);
    if (ret < 0)
      die("av_frame_make_writable: %s\n", av_err2str(ret));
    more = draw(i);
    if (more < 0)
      break;
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, 
        GL_UNSIGNED_BYTE, pixels); 
    ret = sws_scale(sws, &src,
        &src_stride, 0, HEIGHT, 
        frame->data, frame->linesize);
    frame->pts = i++;
    encode(cctx, frame, pkt, fmtctx, vid);
  } while (more > 0);
  encode(cctx, NULL, pkt, fmtctx, vid);
  av_frame_free(&frame);
  av_packet_free(&pkt);
  av_write_trailer(fmtctx);
  avio_close(fmtctx->pb);
  avcodec_free_context(&cctx);
  avformat_free_context(fmtctx);
}

static void discard_line(void) {
  int c;
  do {
    c = getchar();
  } while (c != '\n' && c != EOF);
}

int main(int argc, char **argv) {
  switch (argc) {
  case 0:
  case 1:
    break;
  case 3:
    path = argv[2];
    /* fall through */
  case 2:
    FILE *tmp = freopen(argv[1], "rb", stdin);
    if (!tmp) {
      perror("freopen");
      exit(1);
    }
    stdin = tmp;
    break;
  default:
    fputs("too many args\n", stderr);
    exit(1);
  }
  SDL_Window *wnd;
  GLuint fbo;
  GLuint tex;
  if (SDL_Init(SDL_INIT_EVERYTHING))
    die("SDL_Init: %s\n", SDL_GetError());
  if (atexit(SDL_Quit))
    die("atexit: SDL_Quit\n");
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, 
      SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  wnd = SDL_CreateWindow("Billiards", SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 
      SDL_WINDOW_OPENGL);
  if (!wnd)
    die("SDL_CreateWindow: %s\n", SDL_GetError());
  if (!SDL_GL_CreateContext(wnd))
    die("SDL_GL_CreateContext: %s\n", SDL_GetError());
  gladLoadGL((GLADloadfunc) SDL_GL_GetProcAddress);
  init_draw();
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 
      0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
      GL_TEXTURE_2D, tex, 0);
  glGenerateMipmap(GL_TEXTURE_2D);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != 
      GL_FRAMEBUFFER_COMPLETE)
    die("glCheckFramebufferStatus: %u\n", glGetError());
  discard_line();
  render();
  return 0;
}
