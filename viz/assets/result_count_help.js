export function countHelp(data,el,table){
 const x=data.execution;if(!x)return el('div');
 const detail=el('details',null,'count-help');detail.append(el('summary','这些计数是什么意思？'));
 detail.append(el('p',`${data.summary.total} 个请求 · ${x.input_tokens} 个输入 Token · ${x.generated_tokens} 个输出 Token · ${x.batches} 个执行批次 · ${x.active_gpu_ranks} 个 GPU rank 参与计算。`));
 const methods=Object.entries(x.transfer_methods).map(([k,n])=>`${n} 次 ${k==='protobridge'?'ProtoBridge':k==='flowsim'?'FlowSim':k}`).join(' + ');
 detail.append(table(['计数对象','本次数量','口径'],[
  ['计算 / 存储调用',x.compute_calls,'按批次、GPU 和重复执行累计，包含矩阵运算、向量运算、注意力和 KV 读写；不是模型层数或硬件实测的 kernel 数。'],
  ['通信操作（COMM）',x.communication_calls,'执行图中的集合通信或点对点通信。一项集合通信可以涉及多个 GPU，不能直接当作网络跳数。'],
  ['端到端消息',methods,'请求输入、Token 输出、输入分发等消息的记录；部分由 ProtoBridge 估算，部分由网络后端逐流执行。'],
  ['带宽服务区间',x.service_intervals,'一条消息经多跳转发，每跳的带宽变化还可分成多个区间；数量不等于消息数、请求数或独立链路数。']
 ]));
 detail.append(table(['执行批次','计算 / 存储调用','通信操作'],x.batch_calls.map(b=>[b.id,b.compute,b.communication])));
 const operations={allreduce:'AllReduce：GPU 合并结果',allgather:'AllGather：GPU 汇集分片',sendrecv:'Send/Recv：阶段间发送与接收'};
 detail.append(el('p',Object.entries(x.communication_operations).map(([k,n])=>`${operations[k]||k} ${n} 次`).join('；')));
 detail.append(el('p','计数描述仿真执行工作量，不代表服务时延。并行 GPU 的耗时不能直接相加；ProtoBridge 通信仍保留依赖、资源占用与完成事件。'));
 return detail;
}
