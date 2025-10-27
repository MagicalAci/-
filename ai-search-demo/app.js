/* WikiFX AI 搜索 Demo - P0：意图识别 + 导航代理（本地规则版） */

const CATEGORY_TYPES = [
  "交易商",
  "自营交易商",
  "企业",
  "官方号",
  "内容",
  "用户",
  "区块",
];

const WIKIFX_SEARCH_BASE =
  "https://www.wikifx.com/zh-cn/search.html?SearchPosition=1&scene=1&keyword=";

const KNOWN_BROKERS = [
  {
    name: "XM",
    slug: "xm",
    type: "交易商",
    company: "Trading Point Holdings Ltd",
    regulatedBy: ["ASIC", "CySEC"],
    official: ["微博", "微信"],
    popularIntents: ["出金", "入金", "监管", "点差", "App"],
  },
  {
    name: "IC Markets",
    slug: "ic markets",
    type: "交易商",
    company: "International Capital Markets Pty Ltd",
    regulatedBy: ["ASIC"],
    official: ["Twitter"],
    popularIntents: ["监管", "ECN", "出金", "平台"],
  },
  {
    name: "Exness",
    slug: "exness",
    type: "交易商",
    company: "Exness Group",
    regulatedBy: ["FCA", "CySEC"],
    official: ["Twitter", "Facebook"],
    popularIntents: ["出金", "入金", "点差", "监管"],
  },
];

const KNOWN_COMPANIES = [
  { name: "Trading Point Holdings Ltd", type: "企业" },
  { name: "International Capital Markets Pty Ltd", type: "企业" },
  { name: "Exness Group", type: "企业" },
];

const KNOWN_OFFICIAL = [
  { name: "XM 官方号", platform: "微博", type: "官方号" },
  { name: "IC Markets 官方号", platform: "Twitter", type: "官方号" },
  { name: "Exness 官方号", platform: "Twitter", type: "官方号" },
];

const KNOWN_USERS = [
  { name: "@外汇老兵", type: "用户" },
  { name: "@搬砖小能手", type: "用户" },
];

const KNOWN_CONTENT = [
  { title: "XM 出金教程与常见问题", type: "内容" },
  { title: "IC Markets 是否受 ASIC 严格监管？", type: "内容" },
  { title: "Exness 点差与交易时段一览", type: "内容" },
];

const KNOWN_BLOCK = [
  { name: "USDT-TRC20 热门转账监控", type: "区块" },
  { name: "链上地址归因（示例）", type: "区块" },
];

const SYNONYMS = new Map([
  ["icmarkets", "ic markets"],
  ["ic market", "ic markets"],
  ["艾克马", "ic markets"],
  ["埃克斯尼斯", "exness"],
]);

function normalizeQuery(input) {
  const text = (input || "").trim().toLowerCase();
  return text.replace(/\s+/g, " ");
}

function replaceSynonyms(text) {
  let result = text;
  for (const [k, v] of SYNONYMS.entries()) {
    result = result.replaceAll(k, v);
  }
  return result;
}

function detectEntities(text) {
  const found = [];
  for (const b of KNOWN_BROKERS) {
    const name = b.name.toLowerCase();
    const slug = b.slug.toLowerCase();
    if (text.includes(name) || text.includes(slug)) {
      found.push({ type: b.type, name: b.name, slug: b.slug });
    }
  }
  // company
  for (const c of KNOWN_COMPANIES) {
    if (text.includes(c.name.toLowerCase())) {
      found.push({ type: c.type, name: c.name });
    }
  }
  return found;
}

function detectIntent(text, entities) {
  const q = text;
  const has = (w) => q.includes(w);

  // Compare intent
  if (/对比|比较|哪个好|vs|VS/.test(q)) {
    return { intent: "compare", reason: "包含比较类词汇" };
  }
  // Complaint / exposure
  if (/投诉|维权|被骗|曝光|黑幕|坑/.test(q)) {
    return { intent: "complaint", reason: "包含投诉/曝光类词汇" };
  }
  // Official accounts
  if (/官方号|官微|微博|微信|twitter|facebook|官网/.test(q)) {
    return { intent: "official", reason: "包含官方渠道相关词汇" };
  }
  // Regulation / license
  if (/监管|牌照|许可证|合规/.test(q)) {
    return { intent: "regulation", reason: "包含监管/牌照类词汇" };
  }
  // Deposit/withdraw
  if (/出金|入金|存款|提款|提现/.test(q)) {
    return { intent: "funding", reason: "包含出入金类词汇" };
  }
  // Company
  if (/公司|集团|法人|控股|工商|企业/.test(q)) {
    return { intent: "company", reason: "包含公司/企业类词汇" };
  }
  // Block / onchain
  if (/区块|链上|合约地址|交易哈希|tx/.test(q)) {
    return { intent: "block", reason: "包含区块/链上相关词汇" };
  }
  // If entity detected, default navigate
  if (entities.length > 0) {
    return { intent: "navigate", reason: "识别到实体，缺省为导航" };
  }
  // default content search
  return { intent: "content", reason: "未识别特定意图，缺省为内容检索" };
}

function decideTargetType(intent) {
  switch (intent) {
    case "compare":
    case "regulation":
    case "funding":
    case "navigate":
      return "交易商";
    case "official":
      return "官方号";
    case "company":
      return "企业";
    case "block":
      return "区块";
    case "complaint":
      return "内容";
    default:
      return "内容";
  }
}

function buildWikiFxSearchUrl(keyword) {
  return `${WIKIFX_SEARCH_BASE}${encodeURIComponent(keyword)}`;
}

function computeConfidence(intent, entities) {
  let score = 0.5;
  if (entities.length > 0) score += 0.2;
  if (["compare", "official", "regulation", "funding", "company", "block"].includes(intent)) score += 0.2;
  if (intent === "navigate") score += 0.1;
  return Math.max(0, Math.min(1, score));
}

function aiSearch(queryRaw, scopeTypeFilter) {
  const normalized = normalizeQuery(queryRaw);
  const q = replaceSynonyms(normalized);
  const entities = detectEntities(q);
  const { intent, reason } = detectIntent(q, entities);

  let targetType = decideTargetType(intent);
  if (scopeTypeFilter && CATEGORY_TYPES.includes(scopeTypeFilter)) {
    targetType = scopeTypeFilter;
  }

  let targetEntity = null;
  if (targetType === "交易商" && entities.length > 0) {
    targetEntity = entities.find((e) => e.type === "交易商") || entities[0];
  }

  const confidence = computeConfidence(intent, entities);
  const targetUrl = buildWikiFxSearchUrl(queryRaw);

  const actions = [];
  actions.push({
    label: "打开 WikiFX 搜索",
    url: targetUrl,
  });
  if (targetEntity && targetEntity.slug) {
    actions.push({
      label: `直达交易商：${targetEntity.name}`,
      url: buildWikiFxSearchUrl(targetEntity.slug),
    });
  }

  const suggestions = buildSuggestions(intent, targetType, entities);
  const relatedEntities = buildRelatedEntities(targetType, entities);

  return {
    normalizedQuery: q,
    intent,
    reason,
    targetType,
    targetEntity,
    targetUrl,
    confidence,
    actions,
    suggestions,
    relatedEntities,
  };
}

function buildSuggestions(intent, targetType, entities) {
  const items = [];
  const base = entities[0]?.name || "XM";
  switch (intent) {
    case "compare":
      items.push(`比较 ${base} 和 IC Markets`);
      items.push(`${base} 点差对比`);
      break;
    case "regulation":
      items.push(`${base} 监管牌照`);
      items.push(`${base} 合规风险`);
      break;
    case "funding":
      items.push(`${base} 出金 时间`);
      items.push(`${base} 入金 手续费`);
      break;
    case "official":
      items.push(`${base} 官方号 微博`);
      items.push(`${base} 官网`);
      break;
    case "company":
      items.push(`${base} 母公司`);
      items.push(`${base} 股权结构`);
      break;
    case "block":
      items.push(`${base} 链上转账`);
      items.push(`${base} 地址归因`);
      break;
    default:
      items.push(`${base} 新闻`);
      items.push(`${base} 曝光 投诉`);
  }
  return items;
}

function buildRelatedEntities(targetType, entities) {
  if (targetType === "交易商") {
    return KNOWN_BROKERS.map((b) => ({ type: b.type, name: b.name, slug: b.slug }));
  }
  if (targetType === "企业") {
    return KNOWN_COMPANIES;
  }
  if (targetType === "官方号") {
    return KNOWN_OFFICIAL;
  }
  if (targetType === "用户") {
    return KNOWN_USERS;
  }
  if (targetType === "内容") {
    return KNOWN_CONTENT;
  }
  if (targetType === "区块") {
    return KNOWN_BLOCK;
  }
  return [];
}

function renderResult(model) {
  const resultSection = document.getElementById("resultSection");
  resultSection.classList.remove("hidden");

  const reasoning = document.getElementById("reasoning");
  reasoning.textContent = `识别意图：${model.intent}；目标类别：${model.targetType}。原因：${model.reason}`;

  const confidenceEl = document.getElementById("confidence");
  confidenceEl.textContent = `置信度：${(model.confidence * 100).toFixed(0)}%`;

  const actionsRow = document.getElementById("actionsRow");
  actionsRow.innerHTML = "";
  model.actions.forEach((a) => {
    const btn = document.createElement("a");
    btn.className = "action-btn";
    btn.href = a.url;
    btn.target = "_blank";
    btn.textContent = a.label;
    actionsRow.appendChild(btn);
  });

  const targetCard = document.getElementById("targetCard");
  targetCard.innerHTML = "";
  const p = document.createElement("div");
  p.className = "item";
  const entityText = model.targetEntity ? `（实体：${model.targetEntity.name}）` : "";
  p.innerHTML = `推荐跳转到 <b>${model.targetType}</b> 结果页 ${entityText}：<br/><a href="${model.targetUrl}" target="_blank">${model.targetUrl}</a>`;
  targetCard.appendChild(p);

  const entitiesCard = document.getElementById("entitiesCard");
  entitiesCard.innerHTML = "";
  if (model.relatedEntities.length === 0) {
    entitiesCard.textContent = "无相关实体";
  } else {
    model.relatedEntities.forEach((e) => {
      const el = document.createElement("div");
      el.className = "item";
      const url = buildWikiFxSearchUrl(e.slug || e.name);
      el.innerHTML = `<b>${e.type}</b> · ${e.name} — <a href="${url}" target="_blank">查看</a>`;
      entitiesCard.appendChild(el);
    });
  }

  const suggestionsCard = document.getElementById("suggestionsCard");
  suggestionsCard.innerHTML = "";
  model.suggestions.forEach((s) => {
    const el = document.createElement("div");
    el.className = "item";
    const url = buildWikiFxSearchUrl(s);
    el.innerHTML = `${s} — <a href="${url}" target="_blank">搜索</a>`;
    suggestionsCard.appendChild(el);
  });
}

function attachUI() {
  const input = document.getElementById("queryInput");
  const btn = document.getElementById("searchBtn");
  const chipsRow = document.getElementById("chipsRow");

  let activeType = null;

  chipsRow.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      chipsRow.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      if (activeType === chip.dataset.type) {
        activeType = null; // toggle off
      } else {
        chip.classList.add("active");
        activeType = chip.dataset.type;
      }
    });
  });

  function doSearch() {
    const q = input.value.trim();
    if (!q) return;
    const model = aiSearch(q, activeType);
    renderResult(model);
  }

  btn.addEventListener("click", doSearch);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });
}

window.addEventListener("DOMContentLoaded", attachUI);