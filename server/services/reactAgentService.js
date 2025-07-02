const axios = require('axios');
const OpenAI = require('openai');
const DoctorRAGService = require('./doctorRagService');
const VectorRAGService = require('./vectorRagService');
const ScrapingBeeService = require('./scrapingBeeService');

// Serper API 搜尋服務
class SerperSearchService {
  constructor() {
    this.apiKey = process.env.SERPER_API_KEY;
    this.baseURL = 'https://google.serper.dev/search';
  }

  async search(query) {
    try {
      if (!this.apiKey) {
        console.log('⚠️ Serper API key 未設定，跳過 Google 搜尋');
        return { organic: [], totalResults: 0 };
      }
      const response = await axios.post(this.baseURL, { q: query, num: 5, gl: 'tw', hl: 'zh-tw' }, {
        headers: { 'X-API-KEY': this.apiKey, 'Content-Type': 'application/json' },
        timeout: 10000
      });
      return response.data;
    } catch (error) {
      console.error('Serper 搜尋錯誤:', error.message);
      return { organic: [], totalResults: 0 };
    }
  }
}

// ReAct Agent 服務
class ReactAgentService {
  constructor() {
    this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || 'dummy-key' });
    this.doctorRag = new DoctorRAGService();
    this.vectorRag = new VectorRAGService();
    this.searchService = new SerperSearchService();
    this.scrapingBeeService = new ScrapingBeeService();
  }

  optimizeSearchQuery(userQuery) {
    let optimizedQuery = userQuery;
    if (userQuery.includes('看到幾號') || userQuery.includes('叫號')) {
      optimizedQuery += ' 即時叫號 門診進度';
    }
    if (userQuery.includes('醫師') && !userQuery.includes('高醫')) {
      optimizedQuery += ' 高雄醫學大學附設醫院';
    }
    return optimizedQuery;
  }

  async run(query, history = []) {
    const systemPrompt = `你是一個醫療 ReAct agent，可使用以下工具：\n` +
      `1. doctor_rag：查詢醫師資料庫\n` +
      `2. vector_rag：向量檢索醫師資料\n` +
      `3. web_search：Google 搜尋\n` +
      `4. finish：輸出最終診斷報告。\n` +
      `請依序輸出 Thought、Action、Action Input，獲得 Observation 後再繼續，直到使用 finish。回覆語言為繁體中文。\n` +
      `當你已經有明確答案時，請務必使用 finish action 結束，不要重複回覆。\n` +
      `在輸出 finish action 時，請直接給出最終答案，不要加上「抱歉」、「我將正確地完成這次對話」等多餘語句或開場白，只需輸出醫療資訊本身。`;
      

    // 將 history 轉為 messages，role 不存在時自動補齊
    let msgHistory = [];
    if (Array.isArray(history)) {
      let nextRole = 'user';
      for (const item of history) {
        if (item.role && item.content) {
          msgHistory.push({ role: item.role, content: item.content });
          nextRole = item.role === 'user' ? 'assistant' : 'user';
        } else if (item.query && item.response) {
          msgHistory.push({ role: 'user', content: item.query });
          msgHistory.push({ role: 'assistant', content: item.response });
          nextRole = 'user';
        } else if (item.query) {
          msgHistory.push({ role: 'user', content: item.query });
          nextRole = 'assistant';
        } else if (item.response) {
          msgHistory.push({ role: 'assistant', content: item.response });
          nextRole = 'user';
        }
      }
    }
    const messages = [
      { role: 'system', content: systemPrompt },
      ...msgHistory,
      { role: 'user', content: query }
    ];

    // 新增：記錄 web_search observation
    let webSearchObservations = [];
    let lastWebSearchInput = '';
    let lastUsefulReply = '';

    for (let i = 0; i < 8; i++) {
      const completion = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages,
        max_tokens: 600,
        temperature: 0.3
      });
      const reply = completion.choices[0].message.content.trim();
      console.log(`第${i+1}回合AI回覆:`, reply);
      const actionMatch = reply.match(/Action\s*:\s*(\w+)/i);
      const finishMatch = reply.match(/Finish\s*:\s*([\s\S]*)/i);
      const inputMatch = reply.match(/Action Input\s*:\s*([\s\S]*)/i);
      const action = actionMatch ? actionMatch[1].toLowerCase() : null;
      const actionInput = inputMatch ? inputMatch[1].trim() : '';

      // 記錄有意義的回覆（非 Action/Action Input/Finish 指令且內容長度大於 15）
      if (
        reply.length > 15 &&
        !/^Action:/i.test(reply) &&
        !/^Action Input:/i.test(reply) &&
        !/^Finish:/i.test(reply)
      ) {
        lastUsefulReply = reply;
      }

      // 新增：只有在回覆同時包含網址且內容長度超過 50 字才提前回傳
      if (/https?:\/\//.test(reply) && reply.length > 50) {
        // 自動補全 web_search observation
        let finalText = reply;
        if (webSearchObservations.length > 0) {
          finalText += '\n\n【相關網路連結】\n';
          webSearchObservations.forEach((obs, idx) => {
            try {
              const arr = JSON.parse(obs);
              arr.forEach((item, i) => {
                finalText += `${idx+1}.${i+1} ${item.title}\n${item.link}\n${item.snippet ? '摘要: ' + item.snippet : ''}\n`;
              });
            } catch (e) {
              finalText += obs + '\n';
            }
          });
        }
        return finalText;
      }

      // 新增：只要 AI 回覆內容有明確答案就提前回傳
      if ((/位於|地址|電話|專長|醫師|醫院|科別/.test(reply) && reply.length > 20)) {
        return reply;
      }

      if (action === 'finish' || finishMatch) {
        const finalMatch = reply.match(/Final Answer\s*:\s*([\s\S]*)/i);
        let finalText = '';
        if (finalMatch) {
          finalText = finalMatch[1].trim();
        } else if (finishMatch) {
          finalText = finishMatch[1].trim();
        } else {
          finalText = reply;
        }
        // 如果 finish 回合內容為空，自動補上上一回合有意義的答案
        if ((!finalText || /^Action: finish$/i.test(reply)) && lastUsefulReply) {
          finalText = lastUsefulReply;
        }
        // 自動補全 web_search observation
        if (webSearchObservations.length > 0) {
          finalText += '\n\n【相關網路連結】\n';
          webSearchObservations.forEach((obs, idx) => {
            try {
              const arr = JSON.parse(obs);
              arr.forEach((item, i) => {
                finalText += `${idx+1}.${i+1} ${item.title}\n${item.link}\n${item.snippet ? '摘要: ' + item.snippet : ''}\n`;
              });
            } catch (e) {
              finalText += obs + '\n';
            }
          });
        }
        return finalText;
      }

      let observation = '';
      if (action === 'doctor_rag') {
        const result = await this.doctorRag.searchDoctors(actionInput || query);
        observation = JSON.stringify(result);
      } else if (action === 'vector_rag') {
        const result = await this.vectorRag.searchDoctors(actionInput || query);
        observation = JSON.stringify(result);
      } else if (action === 'web_search') {
        const result = await this.searchService.search(this.optimizeSearchQuery(actionInput || query));
        observation = JSON.stringify(result.organic ? result.organic.slice(0,3) : []);
        // 新增：記錄 web_search observation
        webSearchObservations.push(observation);
        lastWebSearchInput = actionInput || query;
      } else {
        observation = '未知行動';
      }

      messages.push({ role: 'assistant', content: reply });
      messages.push({ role: 'user', content: `Observation: ${observation}` });
    }
    // 如果 8 回合都沒明確答案，但有 lastUsefulReply，直接回傳
    if (lastUsefulReply) {
      return lastUsefulReply;
    }
    return '無法在限制內完成診斷';
  }
}

async function processMedicalQueryReact(query, history = []) {
  const agent = new ReactAgentService();
  const response = await agent.run(query, history);
  return { response };
}

module.exports = { ReactAgentService, processMedicalQueryReact };