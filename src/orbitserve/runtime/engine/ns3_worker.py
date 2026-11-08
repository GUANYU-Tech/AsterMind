"""Native ns-3 point-to-point serialization with causal RPC boundaries.

OrbitServe chooses hops and accounts for propagation. This adapter deliberately
models packet arbitration, not TCP congestion control or IP route discovery.
"""
import ctypes
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import distribution, version
from pathlib import Path

NATIVE = r'''
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
namespace orbitserve_ns3 {
using namespace ns3;
struct Flow {std::string id,key; double total=0, done=0; bool cancelled=false;};
struct Link {
    Ptr<PointToPointNetDevice> device;
    std::deque<std::string> ready;
    std::string current;
    double rate=0, start=0, duration=0, bytes=0;
    bool busy=false;
    bool resourceGate=false;
    std::vector<std::string> pools;
};
struct PacketTrace {std::string flow; double start, finish, logical, wire, offset;};
class Engine {
 public:
    Engine() {Time::SetResolution(Time::PS);}
    std::map<std::string,Flow> flows;
    std::map<std::string,std::unique_ptr<Link>> links;
    std::vector<std::string> completed;
    std::map<std::string,double> poolRates;
    std::map<std::string,bool> poolBusy;
    std::string lastServed;
    double packetsStarted=0, wireBytesStarted=0;
    std::vector<PacketTrace> packetTrace;
    bool recordPackets=false;
    double now() {return Simulator::Now().GetSeconds();}
    Link& link(std::string key) {
        if (!links.count(key)) {
            auto row=std::make_unique<Link>();
            NodeContainer nodes; nodes.Create(2);
            PointToPointHelper helper;
            helper.SetDeviceAttribute("DataRate",DataRateValue(DataRate(1000000000ULL)));
            helper.SetChannelAttribute("Delay",TimeValue(Seconds(0)));
            auto devices=helper.Install(nodes);
            row->device=DynamicCast<PointToPointNetDevice>(devices.Get(0));
            row->device->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&Engine::txEnd,this,key));
            links[key]=std::move(row);
        }
        return *links.at(key);
    }
    static void txEnd(Engine* self,std::string key,Ptr<const Packet> packet) {
        auto& row=self->link(key); auto& flow=self->flows.at(row.current);
        if(self->recordPackets && !row.resourceGate)self->packetTrace.push_back({flow.id,row.start,self->now(),row.bytes,
                                    static_cast<double>(packet->GetSize()),flow.done});
        if (!flow.cancelled) {
            flow.done+=row.bytes;
            if (flow.done>=flow.total-1e-7) {flow.done=flow.total;self->completed.push_back(flow.id);Simulator::Stop();}
            else row.ready.push_back(flow.id);
        }
        row.busy=false;
        for(auto& pool:row.pools) self->poolBusy[pool]=false;
        // PointToPointNetDevice finishes its own state transition after trace callbacks.
        self->lastServed=key;
        Simulator::ScheduleNow(&Engine::pumpAll,self);
    }
    void addPool(std::string key,double rate){poolRates[key]=rate;poolBusy[key]=false;}
    void attachPool(std::string key,std::string pool){link(key).pools.push_back(pool);}
    void pumpAll() {
        // Rotate the first resource considered to avoid starving another GSL.
        for(auto it=links.upper_bound(lastServed);it!=links.end();++it)pump(it->first);
        for(auto it=links.begin();it!=links.end() && it->first<=lastServed;++it)pump(it->first);
    }
    void pump(std::string key) {
        auto& row=link(key);
        if(row.busy || row.rate<=0)return;
        for(auto& pool:row.pools)if(poolBusy[pool])return;
        while(!row.ready.empty()) {
            std::string id=row.ready.front(); row.ready.pop_front();
            auto& flow=flows.at(id);
            if(flow.cancelled || flow.done>=flow.total)continue;
            double logical=std::min(1400.0,flow.total-flow.done);
            uint32_t payload=static_cast<uint32_t>(std::ceil(logical));
            row.current=id; row.bytes=logical;row.start=now();
            double capacity=row.rate;
            for(auto& pool:row.pools){capacity=std::min(capacity,poolRates[pool]);poolBusy[pool]=true;}
            if(row.resourceGate) {
                auto delay=PicoSeconds(static_cast<int64_t>(std::max(1.0,std::ceil(logical*8.0/capacity*1e12))));
                row.duration=delay.GetSeconds(); row.busy=true;
                Simulator::Schedule(delay,&Engine::txEnd,this,key,Create<Packet>(0));
                return;
            }
            auto rate=DataRate(static_cast<uint64_t>(std::max(1.0,capacity)));
            row.device->SetDataRate(rate);
            row.duration=rate.CalculateBytesTxTime(payload+2).GetSeconds();
            row.busy=true;
            row.device->Send(Create<Packet>(payload),row.device->GetBroadcast(),0x0800);
            packetsStarted+=1; wireBytesStarted+=payload+2;
            return;
        }
    }
    void configure(std::string key,double rate) {auto& row=link(key);row.rate=rate;pump(key);}
    void gate(std::string key,double enabled) {link(key).resourceGate=enabled!=0;}
    void start(std::string id,std::string key,double bytes) {
        Flow flow; flow.id=id;flow.key=key;flow.total=bytes;
        flows[id]=flow;link(key).ready.push_back(id);pump(key);
    }
    double progress(std::string id) {
        auto& flow=flows.at(id);auto& row=link(flow.key);
        double amount=flow.done;
        if(row.busy && row.current==id)amount+=row.bytes*std::min(1.0,std::max(0.0,(now()-row.start)/std::max(1e-12,row.duration)));
        return std::min(flow.total,amount);
    }
    void cancel(std::string id){flows.at(id).cancelled=true;}
    void advance(double target) {
        auto ticks=static_cast<int64_t>(std::ceil(target*1e12));
        auto delay=std::max<int64_t>(0,ticks-Simulator::Now().GetTimeStep());
        if(target>now() && delay==0)delay=1;
        auto deadline=Simulator::Schedule(TimeStep(delay),[](){Simulator::Stop();});
        Simulator::Run();Simulator::Remove(deadline);
    }
};
}
'''

BRIDGE = r'''
namespace {
thread_local std::string lastError;
template<class F> int checked(F action) {
    try { action(); lastError.clear(); return 0; }
    catch(const std::exception& error) { lastError=error.what(); return -1; }
    catch(...) { lastError="Unknown native ns-3 error"; return -1; }
}
using Engine=orbitserve_ns3::Engine;
}
extern "C" {
const char* os_error() {return lastError.c_str();}
void* os_create() {
    Engine* engine=nullptr;
    checked([&](){engine=new Engine();});
    return engine;
}
int os_close(Engine* e) {return checked([&](){ns3::Simulator::Destroy();delete e;});}
int os_now(Engine* e,double* value) {return checked([&](){*value=e->now();});}
int os_packetsStarted(Engine* e,double* value) {return checked([&](){*value=e->packetsStarted;});}
int os_wireBytesStarted(Engine* e,double* value) {return checked([&](){*value=e->wireBytesStarted;});}
int os_trace(Engine* e,double value) {e->recordPackets=value!=0;return 0;}
int os_link(Engine* e,const char* key) {return checked([&](){e->link(key);});}
int os_configure(Engine* e,const char* key,double rate) {return checked([&](){e->configure(key,rate);});}
int os_gate(Engine* e,const char* key,double enabled) {return checked([&](){e->gate(key,enabled);});}
int os_addPool(Engine* e,const char* key,double rate) {return checked([&](){e->addPool(key,rate);});}
int os_attachPool(Engine* e,const char* key,const char* pool) {return checked([&](){e->attachPool(key,pool);});}
int os_start(Engine* e,const char* id,const char* key,double bytes) {return checked([&](){e->start(id,key,bytes);});}
int os_cancel(Engine* e,const char* id) {return checked([&](){e->cancel(id);});}
int os_advance(Engine* e,double target) {return checked([&](){e->advance(target);});}
int os_progress(Engine* e,const char* id,double* value) {return checked([&](){*value=e->progress(id);});}
size_t os_completedCount(Engine* e) {return e->completed.size();}
const char* os_completedAt(Engine* e,size_t i) {return e->completed.at(i).c_str();}
int os_clearCompleted(Engine* e) {e->completed.clear();return 0;}
size_t os_packetCount(Engine* e) {return e->packetTrace.size();}
const char* os_packetFlow(Engine* e,size_t i) {return e->packetTrace.at(i).flow.c_str();}
double os_packetValue(Engine* e,size_t i,int field) {
    const auto& p=e->packetTrace.at(i);
    switch(field){case 0:return p.start;case 1:return p.finish;case 2:return p.logical;
                  case 3:return p.wire;default:return p.offset;}
}
int os_clearPackets(Engine* e) {e->packetTrace.clear();return 0;}
}
'''


class NativeEngine:
    """Use the wheel's native libraries through a small compiled C ABI.

    Compiling against the installed headers avoids cppyy/Cling static initializer
    and JIT traps that differ between workstation and CI compiler environments.
    The temporary build belongs to this worker and is removed on close.
    """

    def __init__(self):
        package = distribution('ns3')
        files = list(package.files or ())
        header = next(path for path in files if path.name == 'core-module.h')
        include = package.locate_file(header).parent.parent
        libraries = []
        for module in ('point-to-point', 'network', 'core'):
            matches = [package.locate_file(path) for path in files
                       if path.name.startswith('libns3') and f'-{module}.' in path.name
                       and path.name.endswith(('.so', '.dylib'))]
            if len(matches) != 1:
                raise RuntimeError(f'Expected one installed ns3 {module} library')
            libraries.extend(matches)
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or not shutil.which(compiler[0]):
            raise RuntimeError('Native ns-3 requires a C++ compiler; install build-essential or set CXX')
        self.build = tempfile.TemporaryDirectory(prefix='orbitserve-ns3-')
        root = Path(self.build.name)
        source, target = root / 'adapter.cc', root / 'adapter.so'
        source.write_text(NATIVE + BRIDGE, encoding='utf-8')
        try:
            subprocess.run([
                *compiler, '-std=c++20', '-O2', '-shared', '-fPIC', f'-I{include}',
                str(source), '-o', str(target), *(str(path) for path in libraries),
                *(f'-Wl,-rpath,{path}' for path in sorted({p.parent for p in libraries})),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
            self.lib = ctypes.CDLL(str(target))
        except subprocess.CalledProcessError as error:
            self.build.cleanup()
            raise RuntimeError(f'Native ns-3 adapter compilation failed:\n{error.stderr}') from error
        except BaseException:
            self.build.cleanup()
            raise
        pointer, string, number = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double
        output = ctypes.POINTER(number)
        signatures = {
            'close': [], 'trace': [number], 'now': [output], 'packetsStarted': [output], 'wireBytesStarted': [output], 'link': [string], 'configure': [string, number],
            'addPool': [string, number], 'attachPool': [string, string], 'gate': [string, number],
            'start': [string, string, number], 'cancel': [string], 'advance': [number],
            'progress': [string, output], 'clearCompleted': [], 'clearPackets': [],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, f'os_{name}')
            function.argtypes = [pointer, *arguments]
            function.restype = ctypes.c_int
        self.lib.os_error.argtypes, self.lib.os_error.restype = [], string
        self.lib.os_create.argtypes, self.lib.os_create.restype = [], pointer
        self.lib.os_completedCount.argtypes, self.lib.os_completedCount.restype = [pointer], ctypes.c_size_t
        self.lib.os_completedAt.argtypes, self.lib.os_completedAt.restype = [pointer, ctypes.c_size_t], string
        self.lib.os_packetCount.argtypes, self.lib.os_packetCount.restype = [pointer], ctypes.c_size_t
        self.lib.os_packetFlow.argtypes, self.lib.os_packetFlow.restype = [pointer, ctypes.c_size_t], string
        self.lib.os_packetValue.argtypes, self.lib.os_packetValue.restype = [pointer, ctypes.c_size_t, ctypes.c_int], number
        self.handle = self.lib.os_create()
        if not self.handle:
            self.build.cleanup()
            raise RuntimeError(self.lib.os_error().decode('utf-8'))

    def call(self, name, *arguments):
        if any(isinstance(value, str) and '\0' in value for value in arguments):
            raise ValueError('Native ns-3 identifiers must not contain NUL')
        values = [value.encode('utf-8') if isinstance(value, str) else value for value in arguments]
        if getattr(self.lib, f'os_{name}')(self.handle, *values):
            raise RuntimeError(self.lib.os_error().decode('utf-8'))

    def value(self, name, *arguments):
        result = ctypes.c_double()
        self.call(name, *arguments, ctypes.byref(result))
        return result.value

    def completed(self):
        return [self.lib.os_completedAt(self.handle, i).decode('utf-8')
                for i in range(self.lib.os_completedCount(self.handle))]

    def close(self):
        try:
            self.call('close')
        finally:
            self.build.cleanup()


def main():
    engine = NativeEngine()
    try:
        serve(engine)
    finally:
        engine.close()


def serve(engine):
    initial = json.loads(sys.stdin.readline())
    for key, rate in initial.get('pools', {}).items():
        engine.call('addPool', key, rate)
    for row in initial['links']:
        if row.get('resource_gate'):
            engine.call('gate', row['key'], 1.)
        for pool in row.get('pools', ()):
            engine.call('attachPool', row['key'], pool)
    active = set()
    if initial.get('start_s', 0) > 0:
        engine.call('advance', initial['start_s'])

    def emit(value):
        print(json.dumps({'time_s': engine.value('now'), **value}, allow_nan=False), flush=True)

    emit({'ready': True, 'version': version('ns3'), 'protocol_version': 1,
          'capabilities': ['packet_trace', 'native_counters', 'dynamic_capacity', 'shared_rf_pools', 'resource_gates']})
    for line in sys.stdin:
        try:
            command = json.loads(line)
            op = command['op']
            if op == 'close':
                emit({'closed': True})
                return
            if op == 'register':
                for row in command['links']:
                    engine.call('link', row['key'])
                    for pool in row.get('pools', ()):
                        engine.call('attachPool', row['key'], pool)
                emit({'registered': True})
            elif op == 'trace':
                engine.call('trace', float(bool(command['enabled'])))
                emit({'tracing': bool(command['enabled'])})
            elif op == 'configure':
                for key, rate in command['capacities'].items():
                    engine.call('configure', key, rate)
                emit({'configured': True})
            elif op == 'start':
                engine.call('start', command['id'], command['key'], command['bytes'] / 1_000_000)
                active.add(command['id'])
                emit({'started': command['id']})
            elif op == 'cancel':
                for identity in command['ids']:
                    engine.call('cancel', identity)
                    active.discard(identity)
                emit({'cancelled': command['ids']})
            elif op == 'advance':
                start = engine.value('now')
                before = {i: engine.value('progress', i) for i in active}
                engine.call('advance', float(command['until_s']))
                done = [i for i in engine.completed() if i in active]
                engine.call('clearCompleted')
                progress = {i: max(0.0, engine.value('progress', i) - before[i]) for i in active}
                active.difference_update(done)
                packets=[]
                for i in range(engine.lib.os_packetCount(engine.handle)):
                    values=[engine.lib.os_packetValue(engine.handle,i,field) for field in range(5)]
                    packets.append({'transfer_id':engine.lib.os_packetFlow(engine.handle,i).decode('utf8'),
                                    **dict(zip(('start_s','finish_s','payload_bytes','wire_bytes','offset_bytes'),values))})
                engine.call('clearPackets')
                emit({'completed': sorted(done), 'bytes_sent': progress, 'interval_start_s': start,
                      'packets': packets,
                      'native_counters': {'packets_started': int(engine.value('packetsStarted')),
                                          'wire_bytes_started': int(engine.value('wireBytesStarted'))}})
            else:
                raise ValueError('unknown ns-3 operation')
        except Exception as error:
            emit({'error': f'{type(error).__name__}: {error}'})


if __name__ == '__main__':
    main()
