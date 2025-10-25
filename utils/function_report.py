import sys
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, Dict, Optional, Type, IO
from requests import Response
from logging import Logger

class FunctionReport(AbstractContextManager):
    def __init__(self, logger: Logger | None = None, max_value_length: int = 200):
        self.logger: Logger = logger
        self._frame = None
        self.start_time = datetime.now()
        self.max_value_length = max_value_length

    def _is_enabled(self) -> bool:
        return self.logger is not None

    def _requests_response_deserialize(self, resp: Response) -> Any:
        try:
            return {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": resp.json()
            }
        except Exception:
            return {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": resp.text
            }

    def out_stream_logs(self, stream: IO, locals_dict: Dict[str, Any]):
        stream.write("Function exited. Local variables:\n")
        for k, v in locals_dict.items():
            value = None
            if isinstance(v, Response):
                value = self._requests_response_deserialize(v)
            else:
                value = v
            stream.write(f"  {k}: {str(value)[:self.max_value_length]}\n")
        end_time = datetime.now()
        duration = end_time - self.start_time
        stream.write(f"start time: {self.start_time}\n")
        stream.write(f"end time: {end_time}\n")
        stream.write(f"Execution time: {duration.total_seconds()} seconds\n")

    def on_exit(self, locals_dict: Dict[str, Any]):
        # self.out_stream_logs(sys.stdout, locals_dict)
        if self._is_enabled():
            self.logger.info("Function exited. Local variables:")
            for k, v in locals_dict.items():
                self.logger.info(f"  {k}: {str(v)[:200]}")
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.logger.info(f"start time: {self.start_time}")
            self.logger.info(f"end time: {end_time}")
            self.logger.info(f"Execution time: {duration.total_seconds()} seconds")

    def __enter__(self):
        # 呼び出し元（withブロック側）のフレームを取得
        if not self._is_enabled():
            return self
        self._frame = sys._getframe(1)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb):
        if not self._is_enabled():
            return False

        try:
            # ローカル変数をコピー（参照切り離しのため dict() で複製）
            locals_copy = dict(self._frame.f_locals)
            self.on_exit(locals_copy)
        finally:
            # 参照循環を避ける
            self._frame = None
        return False  # 例外は伝播


def test_function_report():
    # pytest -s -vv src/utils/function_report.py::test_function_report
    with FunctionReport():
        v0 = 10
        v1 = "test"
        v2 = {
            "key": "value"
        }
        v3 = [1, 2, 3]
    print("Function executed. Check the log file for local variables.")

