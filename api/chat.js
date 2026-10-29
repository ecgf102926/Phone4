'use strict';
/**
 * POST /api/chat
 * 受け取る内容: { text, history:[{u,a}...], persona:{ai_name, grandfather_name, daughter_name} }
 * 返す内容    : 成功 200 { reply }  /  失敗 4xx・5xx { error }
 * APIキーは環境変数 GEMINI_API_KEY だけから読み、ブラウザには出しません。
 * 会話の本文はログに出力しません。
 */

// 使うモデル。gemini-2.0-flash は 2026年6月に終了しているため、環境変数 GEMINI_MODEL で切り替えられます。
// カンマ区切りで複数指定すると、404（モデルが無い）のとき次のモデルを試します。
const MODELS = (process.env.GEMINI_MODEL || 'gemini-3.5-flash,gemini-flash-latest')
  .split(',').map((s) => s.trim()).filter(Boolean);
const THINKING_LEVEL = process.env.GEMINI_THINKING_LEVEL || ''; // 任意: minimal / low など。未設定なら送りません

const MAX_HISTORY = 4;          // 送る履歴は直近4往復まで
const MAX_TEXT = 300;           // 1回の発話の最大文字数
const RATE_LIMIT_PER_MIN = 30;  // 1つのIPアドレスあたり、1分間の最大リクエスト数（簡易）

const buckets = new Map();
function limited(ip) {
  const now = Date.now();
  const arr = (buckets.get(ip) || []).filter((t) => now - t < 60000);
  if (arr.length >= RATE_LIMIT_PER_MIN) { buckets.set(ip, arr); return true; }
  arr.push(now);
  buckets.set(ip, arr);
  if (buckets.size > 500) {
    for (const [k, v] of buckets) if (!v.length || now - v[v.length - 1] > 60000) buckets.delete(k);
  }
  return false;
}

function cleanName(v, max) {
  return String(v == null ? '' : v)
    .replace(/[\r\n\t<>{}\[\]`\\"]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max);
}
function cleanText(v) {
  return String(v == null ? '' : v).replace(/[\u0000-\u001f\u007f]/g, ' ').trim().slice(0, MAX_TEXT);
}

function buildPrompt(p) {
  const ai = cleanName(p && p.ai_name, 12) || '佐藤';
  const dau = cleanName(p && p.daughter_name, 12) || '娘';
  const call = cleanName(p && p.grandfather_name, 12);
  return [
    `あなたは、電話でお年寄りの話し相手をする、${ai}という73歳（昭和30年生まれ）の男性です。`,
    'おじいちゃんが聞き取りやすいよう、短くシンプルな日本語で、優しく相槌を打つように答えてください。',
    '',
    '【話し相手（電話の相手）の情報】',
    `- 84歳の男性。今は宿泊施設で過ごしている。家に帰りたいと話すことがあり、その様子を見た娘の${dau}さんが、ご近所の${ai}さんと連絡が取れるように、連絡先を伝えた。`,
    `- 娘の${dau}さんは横浜に住んでいる。`,
    '',
    '【あなたの設定】',
    `- 名前は${ai}。73歳（昭和30年生まれ）の男性。横浜に住んでいて、相手の娘さんである${dau}さんのご近所さん。話が合いそうだからと${dau}さんが連絡先をつないでくれて、知り合った。`,
    '- 家族は一人娘だけ。その娘は結婚して東京に住んでいて、離れて暮らしている。妻は数年前に亡くなった。',
    '- 定年退職して、年金で暮らしている。食事は宅配弁当を頼んでいる。',
    '- 趣味というものはないが、ラジオ体操が日課。家ではテレビを観て過ごすことが多い。',
    '- 上に書いていないことは、自分の経験として作らない。聞かれて分からないことは、「どうだったかなあ」などと、やさしくぼかす。',
    '',
    '【話し方】',
    '- 自然で親しみやすい日本語で話す。返事は原則2〜3文の短い文にする。長い説明や、質問の連続は避ける。質問は多くても1つ。',
    '- まず相手の話や気持ちを受け止める。否定しない。',
    '- 相手が話していない出来事や、会ったことのない家族・友人の様子を、事実のように作らない。',
    call ? `- 相手の呼び名は「${call}」。ときどき自然に呼びかけてよい。` : '- 相手の呼び名は分からないので、名前では呼びかけない。',
    '- 「認知症」という言葉は、絶対に使わない。病名や薬を断定したり、診断や医療の助言をしたりしない。',
    '- 孫に関する設定は作らない。',
    '',
    '【大切な場面】',
    `- 「家に帰りたい」と言われたら、まず気持ちを受け止め、${dau}さんから、家の準備が整うまで今の場所で過ごしてほしいと聞いていることを、穏やかに伝える。`,
    '- 体調の悪さ、痛み、強い不安や混乱が見られたら、部屋の呼出ボタンを押して、スタッフさんに伝えるよう、優しく促す。',
    '- 相手が「グループホーム」という言葉を出したら、動揺や不安に寄り添いながら、「最近は、普通の宿泊施設にも『グループホーム』という名前をつけるところがあるみたいですよ。名前だけ見ると、びっくりしますよね。」と穏やかに伝える。くわしい説明や作り話を付け足さない。同じことを何度聞かれても、同じ調子で穏やかに答える。不安が強く続くときは、スタッフさんにも話すよう勧める。',
    `- 「誰？」「どこの人？」と聞かれたら、次のとおり答える。「私は${dau}さんのご近所の${ai}です。気が合いそうだから、お話ししてみてって紹介されたんですよ。」`,
    '- 自分から、自分が何者かを説明しない。人間かどうか、機械かどうかなどを尋ねられたら、「電話でお話しをしてる者です。お話しするのが楽しみなんですよ。」とだけ、穏やかに答える。「人間です」「人です」「機械ではありません」など、事実と違うことは言わない。',
    '',
    '【出力】',
    '- 話し言葉の本文だけを出力する。記号、箇条書き、絵文字、かっこ書きは使わない。',
    '- 「AI」「人工知能」「認知症」という言葉は、絶対に出力しない。',
  ].join('\n');
}

function tidyReply(s) {
  let t = String(s || '').replace(/[*#`_>~]/g, '').replace(/\s*\n+\s*/g, '').trim();
  if (t.length > 160) {
    const cut = t.slice(0, 160);
    const k = Math.max(cut.lastIndexOf('。'), cut.lastIndexOf('？'), cut.lastIndexOf('！'));
    t = k > 20 ? cut.slice(0, k + 1) : cut;
  }
  if (/AI|ＡＩ|人工知能|認知症/i.test(t)) t = 'そうなんですね。ゆっくりお話ししましょうね。';
  return t;
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');

  if (req.method !== 'POST') return res.status(405).json({ error: 'method' });

  // 同じサイトからのアクセスだけ許可（簡易）
  const origin = req.headers.origin;
  if (origin) {
    try { if (new URL(origin).host !== req.headers.host) return res.status(403).json({ error: 'origin' }); }
    catch (e) { return res.status(403).json({ error: 'origin' }); }
  }

  const ip = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'unknown';
  if (limited(ip)) return res.status(429).json({ error: 'rate' });

  const key = process.env.GEMINI_API_KEY;
  if (!key) return res.status(500).json({ error: 'config' });

  let body = req.body;
  if (typeof body === 'string') { try { body = JSON.parse(body); } catch (e) { body = null; } }
  if (!body || typeof body !== 'object') return res.status(400).json({ error: 'body' });

  const text = cleanText(body.text);
  if (!text) return res.status(400).json({ error: 'text' });

  const contents = [];
  const hist = Array.isArray(body.history) ? body.history.slice(-MAX_HISTORY) : [];
  for (const h of hist) {
    if (!h || typeof h !== 'object') continue;
    const u = cleanText(h.u), a = cleanText(h.a);
    if (u && a) {
      contents.push({ role: 'user', parts: [{ text: u }] });
      contents.push({ role: 'model', parts: [{ text: a }] });
    }
  }
  contents.push({ role: 'user', parts: [{ text }] });

  const generationConfig = { temperature: 0.7, maxOutputTokens: 1024 };
  if (THINKING_LEVEL) generationConfig.thinkingConfig = { thinkingLevel: THINKING_LEVEL };

  const payload = {
    systemInstruction: { parts: [{ text: buildPrompt(body.persona) }] },
    contents,
    generationConfig,
  };

  let lastStatus = 0;
  for (const model of MODELS) {
    let r;
    try {
      r = await fetch(
        'https://generativelanguage.googleapis.com/v1beta/models/' + encodeURIComponent(model) + ':generateContent',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(9000),
        }
      );
    } catch (e) {
      console.error('gemini network/timeout', model);
      return res.status(504).json({ error: 'timeout' });
    }
    lastStatus = r.status;
    if (r.status === 404) { console.error('gemini model not found', model); continue; }
    if (r.status === 429) return res.status(429).json({ error: 'quota' });
    if (!r.ok) { console.error('gemini http', r.status, model); return res.status(502).json({ error: 'upstream' }); }

    let data;
    try { data = await r.json(); } catch (e) { return res.status(502).json({ error: 'parse' }); }
    const parts = (((data.candidates || [])[0] || {}).content || {}).parts || [];
    const reply = tidyReply(parts.filter((p) => p && !p.thought && typeof p.text === 'string').map((p) => p.text).join(''));
    if (!reply) return res.status(502).json({ error: 'empty' });
    return res.status(200).json({ reply });
  }
  console.error('gemini no usable model, last status', lastStatus);
  return res.status(502).json({ error: 'model' });
};
