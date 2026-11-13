// Independent byte-copy capacity: no matrix, routing or production output.
// 256 threads, three 32-KiB slots, shared allocation admitting two CTAs/SM.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1);} } while(0)
constexpr int Threads=256,Panel=32768,Stages=3,Shared=Panel*Stages;
struct Result { unsigned long long cycles,ns; unsigned checksum,smid; };
__device__ unsigned long long timer_ns(){unsigned long long v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;}
__device__ void copy_panel(unsigned char* dst,const unsigned char* src){
  #pragma unroll
  for(int i=0;i<Panel/(Threads*16);++i){
    unsigned off=(i*Threads+threadIdx.x)*16;
    unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(dst+off));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(src+off):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
}
__global__ __launch_bounds__(Threads,2) void async_panel_l2(
    const unsigned char* input,Result* results,int panels,int rounds){
  extern __shared__ __align__(16) unsigned char storage[];
  unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  auto ns=timer_ns();auto cycles=clock64();
  for(int r=0;r<Stages;++r)copy_panel(storage+r*Panel,input+((blockIdx.x*17+r)%panels)*Panel);
  unsigned checksum=0;
  #pragma unroll 1
  for(int r=0;r<rounds;++r){
    if(r<rounds-Stages)asm volatile("cp.async.wait_group 1;":::"memory");
    else asm volatile("cp.async.wait_group 0;":::"memory");
    __syncthreads();
    int slot=r%Stages;
    checksum+=reinterpret_cast<volatile unsigned*>(storage+slot*Panel)[threadIdx.x];
    __syncthreads();
    int next=r+Stages;
    if(next<rounds)copy_panel(storage+slot*Panel,input+((blockIdx.x*17+next)%panels)*Panel);
  }
  auto end=clock64();auto endns=timer_ns();
  if(threadIdx.x==0)results[blockIdx.x]=Result{static_cast<unsigned long long>(end-cycles),endns-ns,checksum,sm};
}
__global__ void fill(unsigned* input,int words){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=blockDim.x*gridDim.x)input[i]=static_cast<unsigned>(i/(Panel/4)+1);
}
int positive(const char* p){char* end;long n=std::strtol(p,&end,10);if(*end||n<=0||n>1000000)std::exit(2);return n;}
int main(int argc,char** argv){
 if(argc!=5)return 2;
 int grid=positive(argv[1]),mib=positive(argv[2]),rounds=positive(argv[3]),trials=positive(argv[4]);
 if((mib!=8&&mib!=32)||(rounds!=256&&rounds!=1024)||trials>9)return 2;
 cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,0));
 if(std::strcmp(prop.name,"NVIDIA H200")||prop.multiProcessorCount!=132)return 2;
 CHECK(cudaFuncSetAttribute(async_panel_l2,cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
 int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,async_panel_l2,Threads,Shared));
 if(resident!=2)return 3;
 int bytes=mib*1024*1024,panels=bytes/Panel;
 unsigned char* input;Result* output;CHECK(cudaMalloc(&input,bytes));CHECK(cudaMalloc(&output,grid*sizeof(Result)));
 fill<<<132,256>>>(reinterpret_cast<unsigned*>(input),bytes/4);CHECK(cudaDeviceSynchronize());
 cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
 std::vector<Result> host(grid);std::vector<float> samples;
 unsigned long long sumcycles=0,sumns=0;
 for(int trial=-2;trial<trials;++trial){
  async_panel_l2<<<grid,Threads,Shared>>>(input,output,panels,1024);CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(begin));async_panel_l2<<<grid,Threads,Shared>>>(input,output,panels,rounds);
  CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));CHECK(cudaGetLastError());
  float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));
  CHECK(cudaMemcpy(host.data(),output,grid*sizeof(Result),cudaMemcpyDeviceToHost));
  unsigned long long cy=0,ns=0;
  for(int block=0;block<grid;++block){
   unsigned expected=0;for(int r=0;r<rounds;++r)expected+=(block*17+r)%panels+1;
   if(host[block].checksum!=expected)return 4;
   cy+=host[block].cycles;ns+=host[block].ns;
  }
  if(trial>=0){samples.push_back(ms);sumcycles+=cy;sumns+=ns;}
 }
 std::printf("{\"grid\":%d,\"backing_mib\":%d,\"rounds\":%d,\"threads\":%d,\"shared_bytes\":%d,\"resident_ctas\":%d,\"logical_bytes\":%llu,\"sum_cycles\":%llu,\"sum_ns\":%llu,\"samples_ms\":[",grid,mib,rounds,Threads,Shared,resident,static_cast<unsigned long long>(grid)*rounds*Panel,sumcycles,sumns);
 for(int i=0;i<trials;++i)std::printf("%s%.9g",i?",":"",samples[i]);
 std::printf("]}\n");
 CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaFree(output));CHECK(cudaFree(input));
}
