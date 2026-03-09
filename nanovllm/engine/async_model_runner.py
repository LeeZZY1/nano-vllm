import threading
from typing import Optional, Any

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config
from multiprocessing.synchronize import Event


class AsyncModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # rank != 0 的子进程通过 mp.Process 直接实例化 ModelRunner
        # AsyncModelRunner 只会在 rank=0 的主进程中被使用
        # 所以这里不需要区分 rank，直接断言
        assert rank == 0, "AsyncModelRunner 只用于主进程 rank=0"
        
        self.inner = ModelRunner(config, rank, event)
        self._result = None
        self._thread: Optional[threading.Thread] = None
        self._exception: Optional[Exception] = None

    def _background_call(self, seqs, is_prefill, num_prefill_tokens, num_decode_tokens):
        """后台线程：真正执行推理，阻塞直到 GPU 完成并返回"""
        # 打印后台线程的 CUDA stream
        stream = torch.cuda.current_stream()
        print(f"[async thread] CUDA stream id={stream.stream_id}, ptr={stream.cuda_stream}")

        try:
            self._result = self.inner.call(
                "run", seqs, is_prefill, num_prefill_tokens, num_decode_tokens
            )
        except Exception as e:
            self._exception = e

    def run_async(self, seqs, is_prefill, num_prefill_tokens, num_decode_tokens):
        """
        非阻塞：把推理任务扔给后台线程，立即返回。
        主线程可以趁机执行 schedule(N+1)，实现 CPU 调度与 GPU 推理重叠。
        """
        self._result = None
        self._exception = None
        self._thread = threading.Thread(
            target=self._background_call,
            args=(seqs, is_prefill, num_prefill_tokens, num_decode_tokens),
            daemon=True,
        )
        self._thread.start()
        # ⭐ 立即返回

    def wait_for_result(self) -> Optional[Any]:
        """
        阻塞等待后台线程完成（流水线唯一同步点），返回 token_ids。
        """
        if self._thread is None:
            return None
        self._thread.join()   # ⭐ 唯一阻塞点
        self._thread = None
        if self._exception is not None:
            raise self._exception
        return self._result

    def call(self, method_name: str, *args, **kwargs):
        """透传给 inner ModelRunner（用于 exit 等管理操作）"""
        return self.inner.call(method_name, *args, **kwargs)

    def exit(self):
        """清理资源"""
        # 等待所有 pending 完成
        while self.pending_results:
            self.wait_for_result()
        
        # 清理 stream
        if self.use_async:
            torch.cuda.synchronize()
            del self.inference_stream
        
        # 清理 model_runner
        self.model_runner.exit()

    def __getattr__(self, name):
        """
        代理其他属性到 model_runner
        保持接口兼容性
        """
        return getattr(self.model_runner, name)
    
