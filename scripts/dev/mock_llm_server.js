// 本地 DeepSeek 兼容 SSE 模拟服务（仅用于 USB 调试期的端到端链路验收，不参与发布）。
// 用法: node scripts/dev/mock_llm_server.js [port]
const http = require('http');

const PORT = Number(process.argv[2] || 8765);
const REPLY = '漂泊者，语音链路已经接通。现在可以正常对话啦！';

const server = http.createServer((req, res) => {
  const url = (req.url || '').split('?')[0];
  const origin = req.headers.origin || '*';
  const cors = {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Credentials': 'true',
  };
  if (req.method === 'OPTIONS') {
    res.writeHead(204, cors);
    res.end();
    return;
  }
  if (url.endsWith('/models')) {
    res.writeHead(200, { ...cors, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data: [{ id: 'deepseek-chat', object: 'model' }] }));
    return;
  }
  if (url.endsWith('/chat/completions')) {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      res.writeHead(200, {
        ...cors,
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });
      const chars = Array.from(REPLY);
      let i = 0;
      const timer = setInterval(() => {
        if (i >= chars.length) {
          res.write('data: [DONE]\n\n');
          clearInterval(timer);
          res.end();
          return;
        }
        const chunk = {
          id: 'chatcmpl-mock',
          object: 'chat.completion.chunk',
          choices: [{ index: 0, delta: { content: chars[i] }, finish_reason: null }],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        i += 1;
      }, 30);
      // 注意：Node 里 req 的 'close' 在请求体读完就会触发（不是断连），
      // 用它会立刻清掉定时器导致一个 chunk 都不发、客户端永久挂起。用 res 的 'close'。
      res.on('close', () => clearInterval(timer));
    });
    return;
  }
  res.writeHead(404, cors);
  res.end('not found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[mock-llm] listening on http://127.0.0.1:${PORT}`);
});
