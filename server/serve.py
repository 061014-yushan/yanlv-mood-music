#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoodTone 本地静态服务器
- 支持 COOP/COEP 与正确的 MIME 类型（.wasm / .mp3 / .json）
- 默认监听 0.0.0.0:8099，方便手机在同一局域网访问摄像头
用法：
    python3 server/serve.py            # 端口 8099
    python3 server/serve.py 8080       # 自定义端口
"""
import http.server, socketserver, os, sys, mimetypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mimetypes.add_type('application/wasm', '.wasm')
mimetypes.add_type('audio/mpeg', '.mp3')
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/webm', '.webm')


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        # 允许跨源隔离（TensorFlow.js 多线程 wasm 需要）
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        # 只打印错误，保持终端干净
        if len(args) > 1 and str(args[1]).startswith(('4', '5')):
            super().log_message(fmt, *args)


class ReuseAddrServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    with ReuseAddrServer(('0.0.0.0', port), Handler) as httpd:
        print(f'MoodTone 已启动：')
        print(f'  本机访问   http://localhost:{port}/')
        # 打印局域网地址，方便手机访问（摄像头需 https 或 localhost，手机建议用 chrome://flags 允许不安全源）
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            print(f'  局域网访问 http://{ip}:{port}/')
        except Exception:
            pass
        print('按 Ctrl+C 停止')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n已停止')


if __name__ == '__main__':
    main()
