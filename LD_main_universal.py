"""
法規分析系統 - 通用版
支援多種 AI 服務：Ollama、OpenAI、Google Gemini
支援本地資料庫與上傳檔案分析
"""

import streamlit as st
from openai import OpenAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
import requests
import re
import pathlib
import json
import pandas as pd
import io
from typing import List, Tuple, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum

# =============================
# 配置類別
# =============================

class AIProvider(Enum):
    """AI 服務提供者"""
    OLLAMA = "Ollama (本地)"
    OPENAI = "OpenAI"
    GEMINI = "Google Gemini"

@dataclass
class Config:
    """系統配置"""
    BASE_DIR: pathlib.Path = pathlib.Path(__file__).parent
    
    # 預設資料夾
    DEFAULT_DATABASES = {
        "國土計畫法規": BASE_DIR / "laws_txt",
        "都市計畫法規": BASE_DIR / "doji_txt",
        "全國法規JSON": BASE_DIR / "mojLawSplitJSON",
    }
    
    # KEYWORDS_TXT: pathlib.Path = BASE_DIR / "test.txt"
    KEYWORDS_TXT: pathlib.Path = BASE_DIR / "MID_National_1030.txt"
    
    # Origin JSON 路徑
    ORIGIN_JSON: pathlib.Path = BASE_DIR / "Origin" / "OriginBook1104.json"
    
    # Taide JSON 路徑
    TAIDE_JSON: pathlib.Path = BASE_DIR / "Taide" / "Taide1105.json"
    
    # 常數
    TXT_SOURCE_LABEL: str = "法規資料庫"
    JSON_SOURCE_LABEL: str = "全國法規資料庫"
    AI_SOURCE_LABEL: str = "AI建議"
    MAX_CTX_CHARS: int = 16384
    
    # Ollama 設定
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_API_KEY: str = "ollama"
    
    # API 設定
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    
    TIMEOUT: int = 120

config = Config()

# =============================
# 正則表達式模式
# =============================

class RegexPatterns:
    """正則表達式模式集合"""
    SIGNALS_RE = re.compile(r"(本法所稱|所稱|稱為|稱之為|係指|是指|指為|指稱|意指|意謂|意即|謂為|謂之|定義|定義如下|概稱|泛指)")
    ARTICLE_HEAD_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百千]+)\s*條", re.M)
    ENUM_ANCHOR_RE = re.compile(r"(名詞定義|定義如下|本法用語[，、,]\s*定義如下|本法用詞[，、,]\s*定義如下|本條用語[，、,]\s*定義如下|本法所稱)")
    LIST_CTX_RE = re.compile(r"(應包括|應載明|下列(?:內容|事項)?|包括下列|應含|應包含)")
    ENUM_HEAD_RE = re.compile(r"^\s*(?:[（(]?[一二三四五六七八九十百千]+[）)]|[一二三四五六七八九十]+、|\d+[、.])\s*")
    
    PAT_ENUM = re.compile(
        r"^(?P<term>[^：:，,；;]{1,30})\s*[：:，,]\s*"
        r"(?:(?:本法所稱|所稱|係指|是指|指稱|指為|謂為|謂之|意指|意謂|意即|稱為|稱之為|概稱|泛指)\s*)?"
        r"(?P<def>.+)"
    )
    PAT_SENT_1 = re.compile(
        r"(?:本法所稱|所稱)\s*(?P<term>[^，,：:；;]{1,30})[，,：:]\s*"
        r"(?:係指|是指|指稱|指為|謂為|謂之|意指|意謂|意即|稱為|稱之為|概稱|泛指)\s*(?P<def>[^。；\n]+)"
    )
    PAT_SENT_2 = re.compile(
        r"^(?P<term>[^：:，,；;]{1,30})\s*"
        r"(?:係指|是指|指稱|指為|謂為|謂之|稱為|稱之為|意指|意謂|意即)\s*(?P<def>[^。；\n]+)"
    )
    
    TERM_SUFFIX = r"(?:用地|地區|區域|用海|保護區|保安區|類別|分區|帶)"
    SEMANTIC_ENUM_RE = re.compile(
        rf"^(?P<term>[^：:，,；;\s]{{1,30}}{TERM_SUFFIX})\s*[：:]\s*(?P<def>.+)"
    )

patterns = RegexPatterns()

# =============================
# 通用 AI 服務管理
# =============================

class UniversalAIService:
    """通用 AI 服務管理類"""
    
    def __init__(self, provider: AIProvider, api_key: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.client: Optional[Union[OpenAI, Any]] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化 AI 客戶端"""
        if self.provider == AIProvider.OLLAMA:
            self.client = OpenAI(
                base_url=f"{config.OLLAMA_BASE_URL}/v1",
                api_key=config.OLLAMA_API_KEY,
                timeout=config.TIMEOUT,
                max_retries=0
            )
        elif self.provider == AIProvider.OPENAI:
            if self.api_key:
                self.client = OpenAI(
                    api_key=self.api_key,
                    timeout=config.TIMEOUT
                )
        elif self.provider == AIProvider.GEMINI:
            if self.api_key and GEMINI_AVAILABLE and genai:
                genai.configure(api_key=self.api_key)  # type: ignore[attr-defined]
                self.client = genai
    
    def get_available_models(self) -> List[str]:
        """取得可用模型列表"""
        try:
            if self.provider == AIProvider.OLLAMA:
                r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if r.status_code == 200:
                    return [m["name"] for m in r.json().get("models", [])]
            elif self.provider == AIProvider.OPENAI:
                if isinstance(self.client, OpenAI):
                    return [m.id for m in self.client.models.list().data]
            elif self.provider == AIProvider.GEMINI:
                if GEMINI_AVAILABLE and genai is not None:
                    return [
                        m.name for m in genai.list_models()  # type: ignore[attr-defined]
                        if "generateContent" in getattr(m, "supported_generation_methods", [])
                    ]
        except Exception as e:
            st.warning(f"無法取得模型列表: {str(e)}")
        return []
    
    def check_service(self) -> bool:
        """檢查服務是否可用"""
        try:
            if self.provider == AIProvider.OLLAMA:
                r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                return r.status_code == 200
            elif self.provider == AIProvider.OPENAI:
                return bool(self.api_key and isinstance(self.client, OpenAI))
            elif self.provider == AIProvider.GEMINI:
                return bool(self.api_key and GEMINI_AVAILABLE and self.client)
        except Exception:
            pass
        return False
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str, stream: bool = False, **kwargs) -> Any:
        """統一的聊天完成介面"""
        if self.provider == AIProvider.OLLAMA or self.provider == AIProvider.OPENAI:
            if not isinstance(self.client, OpenAI):
                raise ValueError("OpenAI client not initialized")
            return self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                stream=stream,
                **kwargs
            )
        elif self.provider == AIProvider.GEMINI:
            if not GEMINI_AVAILABLE or not genai:
                raise ValueError("Gemini not available. Install with: pip install google-generativeai")
            
            # 轉換訊息格式給 Gemini
            gemini_model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
            
            # 將 OpenAI 格式轉換為 Gemini 格式
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"系統指示: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"使用者: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"助理: {content}\n")
            
            full_prompt = "\n".join(prompt_parts)
            
            if stream:
                response = gemini_model.generate_content(full_prompt, stream=True)
                return response
            else:
                response = gemini_model.generate_content(full_prompt)
                return response
        
        return None

# =============================
# 文件處理工具
# =============================

class FileHandler:
    """文件處理工具類"""
    
    @staticmethod
    def safe_truncate_text(text: str, max_chars: int = config.MAX_CTX_CHARS) -> str:
        if text is None:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]
    
    @staticmethod
    def load_keywords(path: pathlib.Path) -> List[str]:
        if not path.exists():
            return []
        lines = [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]
        return list(dict.fromkeys(lines))

    @staticmethod
    def _decode_bytes(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def load_uploaded_keywords(file_obj: Optional[Any]) -> List[str]:
        if not file_obj:
            return []
        try:
            text = FileHandler._decode_bytes(file_obj.getvalue())
        except Exception:
            return []
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return list(dict.fromkeys(lines))
    
    @staticmethod
    def read_txt_files(folder: pathlib.Path, limit: Optional[int] = None):
        """讀取 TXT 檔案"""
        if not folder.exists():
            return []
        paths = sorted(folder.glob("*.txt"))
        if limit is not None:
            paths = paths[:limit]
        
        out = []
        for p in paths:
            name = p.stem
            law = name.split("-", 1)[1] if "-" in name else name
            text = p.read_text(encoding="utf-8-sig")
            out.append((law, text, p))
        return out

    @staticmethod
    def read_uploaded_txt_files(files: Optional[List[Any]], limit: Optional[int] = None):
        documents = []
        if not files:
            return documents
        for uploaded in files:
            if not uploaded:
                continue
            name = pathlib.Path(uploaded.name)
            if name.suffix.lower() != ".txt":
                continue
            try:
                text = FileHandler._decode_bytes(uploaded.getvalue())
            except Exception:
                continue
            law_name = name.stem.split("-", 1)[1] if "-" in name.stem else name.stem
            documents.append((law_name, text, None))
            if limit is not None and len(documents) >= limit:
                break
        return documents
    
    @staticmethod
    def get_available_databases() -> Dict[str, pathlib.Path]:
        """取得可用的法規資料庫"""
        available = {}
        for name, path in config.DEFAULT_DATABASES.items():
            if path.exists():
                txt_files = list(path.glob("*.txt"))
                json_files = list(path.glob("*.json"))
                if txt_files or json_files:
                    available[f"{name} ({len(txt_files)} 個檔案)"] = path
        return available
    
    @staticmethod
    def load_moj_json(dirpath: pathlib.Path) -> List[Dict[str, Any]]:
        """載入 MOJ JSON 檔案（按法規位階排序：先 Law 後 Order）"""
        out = []
        if not dirpath.exists():
            return out
        
        # 優先載入法規 (ChLaw.json)
        law_file = dirpath / "ChLaw.json"
        if law_file.exists():
            try:
                data = json.loads(law_file.read_text(encoding="utf-8-sig"))
                articles = FileHandler._extract_json_articles(data)
                # 標記為法規
                for article in articles:
                    article["source_type"] = "法規"
                    article["priority"] = 1
                out.extend(articles)
            except Exception:
                pass
        
        # 其次載入命令 (ChOrder.json)
        order_file = dirpath / "ChOrder.json"
        if order_file.exists():
            try:
                data = json.loads(order_file.read_text(encoding="utf-8-sig"))
                articles = FileHandler._extract_json_articles(data)
                # 標記為命令
                for article in articles:
                    article["source_type"] = "命令"
                    article["priority"] = 2
                out.extend(articles)
            except Exception:
                pass
        
        return out

    @staticmethod
    def load_uploaded_json_files(files: Optional[List[Any]]) -> List[Dict[str, Any]]:
        articles: List[Dict[str, Any]] = []
        if not files:
            return articles
        for uploaded in files:
            if not uploaded:
                continue
            try:
                content = FileHandler._decode_bytes(uploaded.getvalue())
                data = json.loads(content)
            except Exception:
                continue
            articles.extend(FileHandler._extract_json_articles(data))
        return articles

    @staticmethod
    def _extract_json_articles(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        laws = data.get("Laws") or data.get("laws") or data.get("items") or []
        for law in laws:
            lawname = (law.get("LawName") or law.get("name") or law.get("Law")
                      or law.get("l") or "").strip()
            # 提取法規修訂日期
            law_modified_date = law.get("LawModifiedDate", "")
            arts = (law.get("LawArticles") or law.get("articles")
                    or law.get("Articles") or law.get("cles") or [])
            for a in arts:
                artno = (a.get("ArticleNo") or a.get("no") or a.get("Article")
                         or a.get("icleNo") or "").strip()
                text = (a.get("ArticleContent") or a.get("content")
                        or a.get("ArticleText") or a.get("icleContent") or "").strip()
                if lawname and artno and text:
                    results.append({
                        "law": lawname, 
                        "article": artno, 
                        "text": text,
                        "modified_date": law_modified_date  # 加入修訂日期
                    })
        return results

    @staticmethod
    def prepare_documents(source: Dict[str, Any], limit: Optional[int]) -> List[Tuple[str, str, Optional[pathlib.Path]]]:
        if source.get("mode") == "upload":
            return FileHandler.read_uploaded_txt_files(source.get("files"), limit)
        path: Optional[pathlib.Path] = source.get("path")
        if path is None:
            return []
        return FileHandler.read_txt_files(path, limit)

    @staticmethod
    def load_keywords_from_source(source: Dict[str, Any]) -> List[str]:
        if source.get("mode") == "upload":
            return FileHandler.load_uploaded_keywords(source.get("file"))
        return FileHandler.load_keywords(config.KEYWORDS_TXT)

    @staticmethod
    def load_json_articles_from_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
        if source.get("mode") == "upload":
            return FileHandler.load_uploaded_json_files(source.get("files"))
        path: Optional[pathlib.Path] = source.get("path")
        if path is None:
            return []
        return FileHandler.load_moj_json(path)
    
    @staticmethod
    def _load_term_json(file_path: pathlib.Path, key_name: str) -> Dict[str, str]:
        """通用的名詞對照 JSON 載入方法
        
        Args:
            file_path: JSON 檔案路徑
            key_name: JSON 中的主鍵名稱（如 "原彙編" 或 "Taide"）
        
        Returns:
            Dict[str, str]: {中文名詞: 內容} 的字典
        """
        term_dict = {}
        if not file_path.exists():
            return term_dict
        
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            term_list = data.get(key_name, [])
            for item in term_list:
                term = item.get("中文名詞", "").strip()
                content = item.get("內容", "").strip()
                if term and content:
                    term_dict[term] = content
        except Exception as e:
            st.warning(f"載入 {file_path.name} 失敗: {str(e)}")
        
        return term_dict
    
    @staticmethod
    def load_origin_json() -> Dict[str, str]:
        """載入 Origin JSON 並建立名詞對照字典"""
        return FileHandler._load_term_json(config.ORIGIN_JSON, "原彙編")
    
    @staticmethod
    def load_taide_json() -> Dict[str, str]:
        """載入 Taide JSON 並建立名詞對照字典"""
        return FileHandler._load_term_json(config.TAIDE_JSON, "Taide")

# =============================
# 法規文本分析器
# =============================

class LegalTextAnalyzer:
    """法規文本分析器"""
    
    @staticmethod
    def split_articles(text: str):
        pieces = []
        matches = list(patterns.ARTICLE_HEAD_RE.finditer(text))
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            pieces.append(("第" + m.group(1) + "條", text[start:end].strip()))
        return pieces
    
    @staticmethod
    def parse_enum_block(block: str, allow_semantic: bool):
        out = []
        for seg in re.split(r"[；;\n]+", block):
            seg = seg.strip()
            if not seg:
                continue
            
            m = patterns.PAT_ENUM.match(seg)
            if m:
                out.append((m.group("term").strip(), m.group("def").strip()))
                continue
            
            if allow_semantic:
                m3 = patterns.SEMANTIC_ENUM_RE.match(seg)
                if m3:
                    out.append((m3.group("term").strip(), m3.group("def").strip()))
                    continue
            
            m1 = patterns.PAT_SENT_1.search(seg) or patterns.PAT_SENT_2.search(seg)
            if m1:
                out.append((m1.group("term").strip(), m1.group("def").strip()))
        return out
    
    @staticmethod
    def extract_candidates(article_text: str):
        cands = []
        head = article_text[:200]
        has_anchor = bool(patterns.ENUM_ANCHOR_RE.search(head))
        has_list_head = bool(patterns.LIST_CTX_RE.search(head))

        lines = [ln.rstrip() for ln in article_text.splitlines()]
        semantic_hint = False
        for ln in lines:
            if patterns.ENUM_HEAD_RE.match(ln):
                content = patterns.ENUM_HEAD_RE.sub("", ln).strip()
                if patterns.SEMANTIC_ENUM_RE.match(content):
                    semantic_hint = True
                    break

        enable_enum = (has_anchor and not has_list_head) or semantic_hint

        if enable_enum:
            buf = []
            for ln in lines:
                if patterns.ENUM_HEAD_RE.match(ln):
                    if buf:
                        cands.extend(LegalTextAnalyzer.parse_enum_block("\n".join(buf), allow_semantic=True))
                        buf = []
                    buf.append(patterns.ENUM_HEAD_RE.sub("", ln))
                else:
                    if buf:
                        buf[-1] += (" " + ln.strip())
            if buf:
                cands.extend(LegalTextAnalyzer.parse_enum_block("\n".join(buf), allow_semantic=True))

        for ln in lines:
            m1 = patterns.PAT_SENT_1.search(ln)
            if m1:
                cands.append((m1.group("term").strip(), m1.group("def").strip()))
            m2 = patterns.PAT_SENT_2.search(ln)
            if m2:
                cands.append((m2.group("term").strip(), m2.group("def").strip()))

        for m in patterns.PAT_SENT_1.finditer(article_text):
            cands.append((m.group("term").strip(), m.group("def").strip()))
        for m in patterns.PAT_SENT_2.finditer(article_text):
            cands.append((m.group("term").strip(), m.group("def").strip()))

        tmp = []
        for t, d in cands:
            if not (1 <= len(t) <= 30):
                continue
            if re.fullmatch(r"[0-9一二三四五六七八九十百千]+", t):
                continue
            tmp.append((t, d.strip().rstrip("。；;")))

        cleaned = []
        for t, d in dict.fromkeys(tmp):
            pos = article_text.find(d)
            window = article_text[max(0, pos - 50):pos + len(d) + 50] if pos != -1 else ""
            if window and patterns.LIST_CTX_RE.search(window):
                continue
            cleaned.append((t, d))
        
        return cleaned

# =============================
# LLM 處理器（通用版）
# =============================

class UniversalLLMProcessor:
    """通用 LLM 處理器"""
    
    def __init__(self, ai_service: UniversalAIService):
        self.ai_service = ai_service
        self.ollama_options = {
            "num_ctx": 8192,
            "num_batch": 512,
            "flash_attention": True
        }
    
    def validate_term_definition(self, term: str, definition: str, article_text: str, model: str) -> bool:
        """LLM 驗證名詞定義"""
        context = FileHandler.safe_truncate_text(article_text, 2000)

        system_prompt = (
            "你是台灣法律文件分析助手。只能使用 <上下文> 的文字判斷，不得改動任何字詞。"
            "任務：檢查 candidates 是否為『名詞—定義』，回傳原文子串。"
            "輸出唯一 JSON：{\"results\":[{\"term\":\"\",\"definition\":\"\",\"defined\":true/false}]}"
            "僅檢核與微調邊界；不要任意新增。"
            "只有『名詞：……』或含『係指/指/謂/為/稱為/意指/意即』，或『X用地：供…使用者』語義枚舉，且不在『應包括/下列內容/應載明/應包含』清單語境，才標 true。"
        )
        user_prompt = (
            f"<上下文>\n{context}\n</上下文>\n\n"
            f"請檢查下列候選是否為有效的「名詞—定義」：\n"
            f"- 名詞：{term}\n"
            f"- 定義候選：{definition}\n\n"
            "請依指示輸出唯一 JSON。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            if self.ai_service.provider == AIProvider.GEMINI:
                response = self.ai_service.chat_completion(messages, model, stream=False)
                if response and hasattr(response, 'text'):
                    result = response.text
                else:
                    st.warning("驗證失敗：無法取得 Gemini 回應")
                    return True
            else:
                extra_params = {}
                if self.ai_service.provider == AIProvider.OLLAMA:
                    extra_params["extra_body"] = {"options": self.ollama_options}
                
                response = self.ai_service.chat_completion(
                    messages, model,
                    stream=False,
                    timeout=config.TIMEOUT,
                    **extra_params
                )
                if response and hasattr(response, 'choices') and response.choices:
                    result = response.choices[0].message.content or ""
                else:
                    st.warning("驗證失敗：無法取得模型回應")
                    return True
            
            try:
                data = json.loads(result)
                first = data.get("results", [{}])[0]
                return bool(first.get("defined"))
            except Exception:
                st.warning(f"驗證失敗：回傳格式錯誤 -> {result}")
                return True
        except Exception as e:
            st.warning(f"驗證失敗：{str(e)}")
            return True
    
    def synthesize_definition(self, term: str, contexts: List[Dict], model: str) -> Tuple[str, str]:
        """使用 AI 合成名詞定義，並回傳定義與上下文摘要"""
        context_texts = "\n\n---\n\n".join([c.get("text", "") for c in contexts[:3]]) if contexts else ""
        truncated_contexts = FileHandler.safe_truncate_text(context_texts, 4000) if context_texts else ""
        
        if truncated_contexts:
            prompt = f"""請仔細閱讀以下法規內容，針對名詞「{term}」整理出清楚的定義：

{truncated_contexts}

請根據上述條文，給出「{term}」的定義，並保持用字專業精準。"""
        else:
            prompt = f"""在未找到對應條文的情況下，請依專業知識推測「{term}」可能的定義，並加入簡短前言註明來源為 AI 合成建議。"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if self.ai_service.provider == AIProvider.GEMINI:
                response = self.ai_service.chat_completion(messages, model, stream=False)
                if response and hasattr(response, 'text'):
                    definition = response.text
                else:
                    definition = "（AI 無法生成定義）"
            else:
                extra_params = {}
                if self.ai_service.provider == AIProvider.OLLAMA:
                    extra_params["extra_body"] = {"options": self.ollama_options}
                
                response = self.ai_service.chat_completion(
                    messages, model,
                    stream=False,
                    timeout=config.TIMEOUT,
                    **extra_params
                )
                if response and hasattr(response, 'choices') and response.choices:
                    definition = response.choices[0].message.content or "（AI 無法生成定義）"
                else:
                    definition = "（AI 無法生成定義）"
        except Exception as e:
            definition = f"（AI 生成失敗: {str(e)}）"
        
        context_summary = truncated_contexts if truncated_contexts else "AI 合成建議：未在資料庫找到對應條文"
        return definition, context_summary

# =============================
# 下載處理器
# =============================

class DownloadHandler:
    """下載處理工具類"""
    
    @staticmethod
    def to_downloadable_excel(df: pd.DataFrame, filename: str = "analysis_results.xlsx"):
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name='分析結果')
                workbook = writer.book
                if hasattr(workbook, "set_properties"):
                    workbook.set_properties({'title': filename})  # type: ignore[attr-defined]
                
                worksheet = writer.sheets['分析結果']
                for i, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    worksheet.set_column(i, i, min(max_length + 2, 50))
            
            buf.seek(0)
            return buf
        except Exception as e:
            st.error(f"產生Excel檔案時發生錯誤: {str(e)}")
            return None
    
    @staticmethod
    def to_downloadable_csv(df: pd.DataFrame, filename: str = "analysis_results.csv"):
        try:
            csv_string = df.to_csv(index=False, encoding='utf-8-sig')
            return csv_string.encode('utf-8-sig')
        except Exception as e:
            st.error(f"產生CSV檔案時發生錯誤: {str(e)}")
            return None

    @staticmethod
    def build_export_filenames(prefix: str, model: str) -> Tuple[str, str]:
        safe_model_name = model.replace(':', '_').replace('/', '_').replace('\\', '_')
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{safe_model_name}_{timestamp}"
        return f"{base}.xlsx", f"{base}.csv"

# =============================
# UI 類別
# =============================

class StreamlitUI:
    """Streamlit UI 工具類"""
    
    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="法規名詞定義分析系統",
            page_icon="⚖️",
            layout="wide"
        )
        st.title("⚖️ 法規名詞定義分析系統")
        st.caption("支援 Ollama、OpenAI、Gemini | 本地資料庫與檔案上傳")
    
    @staticmethod
    def render_sidebar():
        """渲染側邊欄 - 返回 (ai_service, model, mode, dataset_source, keyword_source, use_stream)"""
        with st.sidebar:
            st.header("🛠️ 系統設定")
            if "openai_api_key" not in st.session_state:
                st.session_state["openai_api_key"] = config.OPENAI_API_KEY
            if "gemini_api_key" not in st.session_state:
                st.session_state["gemini_api_key"] = config.GEMINI_API_KEY
            
            # AI 服務選擇
            st.subheader("🤖 AI 服務")
            
            # 檢查 Gemini 是否可用
            available_providers = [AIProvider.OLLAMA.value, AIProvider.OPENAI.value]
            if GEMINI_AVAILABLE:
                available_providers.append(AIProvider.GEMINI.value)
            else:
                st.caption("⚠️ Gemini 不可用（需安裝 google-generativeai）")
            
            provider_choice = st.selectbox(
                "選擇 AI 服務",
                options=available_providers,
                index=0
            )
            
            # API Key 輸入
            api_key = ""
            if provider_choice == AIProvider.OPENAI.value:
                stored_key = st.session_state.get("openai_api_key", config.OPENAI_API_KEY)
                if not isinstance(stored_key, str):
                    stored_key = config.OPENAI_API_KEY
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=stored_key,
                    help="請輸入您的 OpenAI API Key"
                )
                if api_key != stored_key:
                    st.session_state["openai_api_key"] = api_key
            elif provider_choice == AIProvider.GEMINI.value:
                stored_key = st.session_state.get("gemini_api_key", config.GEMINI_API_KEY)
                if not isinstance(stored_key, str):
                    stored_key = config.GEMINI_API_KEY
                api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=stored_key,
                    help="請輸入您的 Google Gemini API Key"
                )
                if api_key != stored_key:
                    st.session_state["gemini_api_key"] = api_key
            
            # 建立 AI 服務
            provider_enum = AIProvider.OLLAMA
            for p in AIProvider:
                if p.value == provider_choice:
                    provider_enum = p
                    break
            
            ai_service = UniversalAIService(provider_enum, api_key)
            
            # 檢查服務狀態
            service_ok = ai_service.check_service()
            if service_ok:
                st.success(f"✅ {provider_choice} 服務正常")
            else:
                st.error(f"❌ {provider_choice} 服務無法連接")
                if provider_enum == AIProvider.OLLAMA:
                    st.info("請確保 Ollama 服務正在運行")
                else:
                    st.info("請檢查 API Key 是否正確")
            
            # 模型選擇
            st.subheader("🎯 模型選擇")
            available_models = ai_service.get_available_models()
            
            if available_models:
                selected_model = st.selectbox(
                    "選擇模型",
                    options=available_models,
                    help=f"從 {provider_choice} 選擇可用的模型"
                )
            else:
                st.warning(f"⚠️ 無法取得 {provider_choice} 模型列表")
                selected_model = st.text_input("手動輸入模型名稱", value="gpt-3.5-turbo")
            
            # 法規資料庫選擇
            st.subheader("📚 法規資料庫")
            database_mode = st.radio(
                "資料來源",
                options=["使用本地資料夾", "上傳 TXT 檔"],
                key="database_mode"
            )

            dataset_source: Dict[str, Any]

            if database_mode == "使用本地資料夾":
                available_dbs = FileHandler.get_available_databases()
                if available_dbs:
                    selected_db_name = st.selectbox(
                        "選擇法規資料庫",
                        options=list(available_dbs.keys()),
                        help="選擇要分析的法規資料庫"
                    )
                    database_path = available_dbs[selected_db_name]
                    txt_files = list(database_path.glob("*.txt"))
                    st.info(f"📁 資料夾: {database_path.name}\n\n📄 檔案數: {len(txt_files)}")
                else:
                    st.error("❌ 找不到可用的法規資料庫")
                    database_path = config.DEFAULT_DATABASES.get("國土計畫法規", config.BASE_DIR)
                dataset_source = {"mode": "local", "path": database_path}
            else:
                uploaded_files = st.file_uploader(
                    "上傳一個或多個 TXT 檔案",
                    type=["txt"],
                    accept_multiple_files=True,
                    key="uploaded_db_files"
                )
                file_count = len(uploaded_files or [])
                st.info(f"📄 已選擇 {file_count} 個檔案")
                dataset_source = {"mode": "upload", "files": uploaded_files}

            # 關鍵字來源
            st.subheader("🔑 關鍵字列表")
            keyword_mode = st.radio(
                "關鍵字來源",
                options=["使用預設檔案", "上傳 TXT"],
                key="keyword_mode"
            )

            keyword_source: Dict[str, Any]
            if keyword_mode == "上傳 TXT":
                keyword_file = st.file_uploader(
                    "上傳關鍵字 TXT 檔案",
                    type=["txt"],
                    key="keyword_upload"
                )
                keyword_source = {"mode": "upload", "file": keyword_file}
                if keyword_file:
                    st.caption(f"🔤 關鍵字檔案：{keyword_file.name}")
            else:
                keyword_source = {"mode": "local"}
                st.caption(f"使用預設檔案：{config.KEYWORDS_TXT.name}")
            
            # 模式選擇
            st.subheader("⚙️ 分析模式")
            mode = st.radio(
                "選擇模式",
                options=["法規分析", "AI 聊天"],
                index=0
            )
            
            # 串流模式設定（僅在 AI 聊天模式顯示）
            use_stream = True
            if mode == "AI 聊天":
                st.subheader("🔧 聊天設定")
                use_stream = st.checkbox(
                    "啟用串流模式",
                    value=True,
                    help="串流模式：逐字顯示回應（需組織驗證）\n非串流模式：一次顯示完整回應"
                )
                if not use_stream:
                    st.info("💡 使用非串流模式（適用於未驗證的 OpenAI 組織）")
            
            st.divider()
            st.caption("💡 提示：")
            st.caption("• Ollama：地端，免費")
            st.caption("• OpenAI：需要 API Key")
            st.caption("• Gemini：需要 API Key")
            if provider_choice == AIProvider.OPENAI.value:
                st.caption("• 串流模式需組織驗證")
        
        return ai_service, selected_model, mode, dataset_source, keyword_source, use_stream

    @staticmethod
    def display_results(rows: List[Dict]) -> Optional[pd.DataFrame]:
        """顯示分析結果"""
        if not rows:
            st.warning("⚠️ 沒有找到任何名詞定義")
            return None
        
        # 統計資訊
        st.subheader("📊 分析結果統計")
        df = pd.DataFrame(rows)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總計名詞", len(df))
        with col2:
            keyword_count = len(df[df["主題詞"] == "是"]) if "主題詞" in df.columns else 0
            st.metric("主題詞", keyword_count)
        with col3:
            has_def_count = len(df[df["有無定義"] == "有"]) if "有無定義" in df.columns else len(df)
            st.metric("有定義", has_def_count)
        with col4:
            unique_sources = df["定義來源"].nunique() if "定義來源" in df.columns else 1
            st.metric("來源數", unique_sources)
        
        # 顯示結果表格
        st.subheader("📋 詳細結果")
        st.dataframe(df, use_container_width=True, height=400)
        
        # 分組統計
        if "定義來源" in df.columns:
            st.subheader("📈 來源分佈")
            source_counts = df["定義來源"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(source_counts)
            with col2:
                for source, count in source_counts.items():
                    st.write(f"**{source}**: {count} 個")
        
        return df

# =============================
# 法規分析引擎
# =============================

class LegalAnalysisEngine:
    """法規分析引擎"""
    
    def __init__(self, llm_processor: UniversalLLMProcessor):
        self.llm = llm_processor
        self.analyzer = LegalTextAnalyzer()
        # 載入 Origin JSON 字典
        self.origin_dict = FileHandler.load_origin_json()
        # 載入 Taide JSON 字典
        self.taide_dict = FileHandler.load_taide_json()
    
    def check_origin_term(self, term: str) -> Tuple[str, str]:
        """檢查名詞是否存在於 Origin JSON 中
        
        Returns:
            Tuple[str, str]: (是否存在("是"/"否"), 原彙編定義內容)
        """
        if term in self.origin_dict:
            return "是", self.origin_dict[term]
        return "否", ""
    
    def check_taide_term(self, term: str) -> Tuple[str, str]:
        """檢查名詞是否存在於 Taide JSON 中
        
        Returns:
            Tuple[str, str]: (是否存在("是"/"否"), Taide定義內容)
        """
        if term in self.taide_dict:
            return "是", self.taide_dict[term]
        return "否", ""
    
    def analyze_full(
        self,
        dataset_source: Dict[str, Any],
        keyword_source: Dict[str, Any],
        limit_files: Optional[int],
        use_llm_validation: bool,
        include_json_search: bool,
        model: str,
    ) -> List[Dict]:
        """完整分析（原版邏輯）"""
        txt_items = FileHandler.prepare_documents(dataset_source, limit_files)

        keywords = set(FileHandler.load_keywords_from_source(keyword_source))

        json_articles: List[Dict[str, Any]] = []
        if include_json_search:
            json_source: Dict[str, Any] = {
                "mode": "local",
                "path": config.DEFAULT_DATABASES.get("全國法規JSON"),
            }
            json_articles = FileHandler.load_json_articles_from_source(json_source)
        
        if not txt_items:
            st.warning("沒有找到任何法規檔案")
            return []
        
        rows = []
        extracted_terms = set()
        
        total_steps = len(txt_items) + (len(keywords) if include_json_search else 0)
        current_step = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # A) 從 TXT 抽取
        status_text.text("📄 分析 TXT 法規檔案...")
        for law, full_text, _ in txt_items:
            status_text.text(f"正在分析: {law}")
            
            articles = self.analyzer.split_articles(full_text)
            
            for art_no, art_txt in articles:
                # 檢查是否有定義信號
                if not patterns.SIGNALS_RE.search(art_txt):
                    continue
                
                # 抽取候選名詞
                candidates = self.analyzer.extract_candidates(art_txt)
                
                # LLM 驗證（可選）
                if use_llm_validation:
                    validated_results = []
                    for term, definition in candidates:
                        if self.llm.validate_term_definition(term, definition, art_txt, model):
                            validated_results.append({"term": term, "definition": definition})
                    results = validated_results
                else:
                    results = [{"term": t, "definition": d} for t, d in candidates]
                
                # 記錄結果
                for r in results:
                    term = r["term"]
                    extracted_terms.add(term)
                    
                    # 檢查是否存在於 Origin JSON
                    origin_exists, origin_content = self.check_origin_term(term)
                    
                    # 檢查是否存在於 Taide JSON
                    taide_exists, taide_content = self.check_taide_term(term)
                    
                    rows.append({
                        "名詞": term,
                        "主題詞": "是" if term in keywords else "否",
                        "有無定義": "有",
                        "定義來源": config.TXT_SOURCE_LABEL,
                        "法規來源": f"{law} {art_no}",
                        "定義": r["definition"],
                        "來源依據(上下文)": art_txt,
                        "原彙編詞": origin_exists,
                        "原彙編定義": origin_content,
                        "Taide詞": taide_exists,
                        "Taide定義": taide_content
                    })
            
            current_step += 1
            progress_bar.progress(min(1.0, current_step / max(1, total_steps)))
        
        # B) 主題字補查（從 JSON 資料庫）
        if include_json_search and keywords and json_articles:
            status_text.text("🔍 主題字補查...")
            for kw in sorted(keywords):
                if kw in extracted_terms:
                    continue
                
                status_text.text(f"補查主題字: {kw}")
                rec = self._search_json_for_definition(kw, json_articles, model)
                
                # 判斷定義是否有效：
                # 1. 定義內容存在且不是錯誤訊息
                # 2. 必須有上下文（從 JSON 資料庫找到相關條文）
                definition_content = rec.get("定義", "")
                has_context = rec.get("has_context", True)  # 從 JSON 抽取的一定有上下文
                has_valid_definition = (
                    bool(definition_content and not definition_content.startswith("❌"))
                    and has_context  # 必須有上下文才算有定義
                )
                
                # 檢查是否存在於 Origin JSON
                origin_exists, origin_content = self.check_origin_term(kw)
                
                # 檢查是否存在於 Taide JSON
                taide_exists, taide_content = self.check_taide_term(kw)
                
                rows.append({
                    "名詞": rec["名詞"],
                    "主題詞": "是",
                    "有無定義": "有" if has_valid_definition else "無",
                    "定義來源": rec.get("定義來源", config.AI_SOURCE_LABEL),
                    "法規來源": rec.get("法規來源", ""),
                    "定義": definition_content,
                    "來源依據(上下文)": rec.get("來源依據(上下文)", ""),
                    "原彙編詞": origin_exists,
                    "原彙編定義": origin_content,
                    "Taide詞": taide_exists,
                    "Taide定義": taide_content
                })
                
                current_step += 1
                progress_bar.progress(min(1.0, current_step / max(1, total_steps)))
        
        status_text.text("✅ 分析完成！")
        return rows
    
    def _search_json_for_definition(self, term: str, json_articles: List[Dict], model: str) -> Dict:
        """在 JSON 資料庫中搜尋名詞定義（優先法規，次之命令；同位階則優先新法）。找不到則由 AI 合成。"""
        found_contexts_law = []  # 法規來源
        found_contexts_order = []  # 命令來源
        
        # 搜尋包含該名詞的條文，並按來源類型分類
        for article in json_articles:
            text = article.get("text", "")
            if term in text and patterns.SIGNALS_RE.search(text):
                context = {
                    "law": article.get("law", ""),
                    "article": article.get("article", ""),
                    "text": text,
                    "source_type": article.get("source_type", "未知"),
                    "priority": article.get("priority", 99),
                    "modified_date": article.get("modified_date", "")  # 加入修訂日期
                }
                
                # 根據來源類型分類
                if article.get("priority", 99) == 1:  # 法規
                    found_contexts_law.append(context)
                else:  # 命令或其他
                    found_contexts_order.append(context)
        
        # 在各自類別內，按修訂日期排序（新的在前）
        found_contexts_law.sort(key=lambda x: x.get("modified_date", ""), reverse=True)
        found_contexts_order.sort(key=lambda x: x.get("modified_date", ""), reverse=True)
        
        # 優先使用法規的定義
        found_contexts = found_contexts_law if found_contexts_law else found_contexts_order
        
        # 如果找到，提取定義
        if found_contexts:
            best_match = found_contexts[0]
            candidates = self.analyzer.extract_candidates(best_match["text"])
            
            for t, d in candidates:
                if t == term:
                    modified_date = best_match.get("modified_date", "")
                    date_display = f" ({modified_date})" if modified_date else ""
                    source_label = f"【{best_match.get('source_type', '未知')}】{best_match['law']} {best_match['article']}{date_display}"
                    return {
                        "名詞": term,
                        "定義來源": config.JSON_SOURCE_LABEL,
                        "法規來源": source_label,
                        "定義": d,
                        "來源依據(上下文)": "\n\n---\n\n".join([c["text"] for c in found_contexts[:3]]),
                        "has_context": True  # 從 JSON 找到的一定有上下文
                    }
        
        # 若未找到符合定義的條文，嘗試放寬條件收集上下文（仍優先法規）
        if not found_contexts:
            relaxed_law = []
            relaxed_order = []
            
            for article in json_articles:
                text = article.get("text", "")
                if term in text:
                    context = {
                        "law": article.get("law", ""),
                        "article": article.get("article", ""),
                        "text": text,
                        "source_type": article.get("source_type", "未知"),
                        "priority": article.get("priority", 99),
                        "modified_date": article.get("modified_date", "")
                    }
                    
                    if article.get("priority", 99) == 1:
                        relaxed_law.append(context)
                    else:
                        relaxed_order.append(context)
            
            # 各類別內按修訂日期排序（由新到舊）
            relaxed_law.sort(key=lambda x: x.get("modified_date", ""), reverse=True)
            relaxed_order.sort(key=lambda x: x.get("modified_date", ""), reverse=True)
            
            # 優先使用法規的上下文，再使用命令
            found_contexts = (relaxed_law + relaxed_order)[:3]
        
        # 未找到，使用 AI 合成
        synth_def, synth_context = self.llm.synthesize_definition(term, found_contexts, model)
        suggested_source = "AI 合成建議"
        if found_contexts:
            first_ctx = found_contexts[0]
            source_type = first_ctx.get('source_type', '未知')
            suggested_source = f"【{source_type}】{first_ctx.get('law', '')} {first_ctx.get('article', '')}".strip()
        
        return {
            "名詞": term,
            "定義來源": config.AI_SOURCE_LABEL,
            "法規來源": suggested_source,
            "定義": synth_def,
            "來源依據(上下文)": synth_context,
            "has_context": bool(found_contexts)  # 標記是否有上下文
        }

# =============================
# 主程式
# =============================

def main():
    """主程式"""
    StreamlitUI.setup_page()
    (
        ai_service,
        selected_model,
        mode,
        dataset_source,
        keyword_source,
        use_stream,
    ) = StreamlitUI.render_sidebar()
    
    llm_processor = UniversalLLMProcessor(ai_service)
    analysis_engine = LegalAnalysisEngine(llm_processor)

    def build_dataset_info(source: Dict[str, Any]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "mode": source.get("mode", "local"),
            "has_documents": False,
            "count": 0,
        }
        if info["mode"] == "upload":
            files = source.get("files") or []
            info["files"] = files
            info["count"] = len(files)
            info["has_documents"] = len(files) > 0
            if files:
                preview = ", ".join(f.name for f in files[:3])
                if len(files) > 3:
                    preview += " ..."
                info["preview"] = preview
            return info

        path = source.get("path")
        info["path"] = path
        if isinstance(path, pathlib.Path) and path.exists():
            txt_files = list(path.glob("*.txt"))
            info["txt_files"] = txt_files
            info["count"] = len(txt_files)
            info["has_documents"] = len(txt_files) > 0
        return info

    dataset_info = build_dataset_info(dataset_source)
    
    if mode == "法規分析":
        st.header("📋 法規名詞定義抽取")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📁 資料庫狀態")
            if dataset_info["mode"] == "upload":
                files = dataset_info.get("files", [])
                st.write("資料來源：上傳 TXT 檔案")
                st.write(f"📄 上傳檔案數: {dataset_info.get('count', 0)}")
                if dataset_info.get("preview"):
                    st.caption(dataset_info["preview"])
            else:
                st.write("資料來源：本地資料夾")
                st.write(f"📄 法規資料庫: {'✅' if dataset_info.get('has_documents') else '❌'}")
                if dataset_info.get("has_documents"):
                    st.write(f"找到 {dataset_info.get('count', 0)} 個法規檔案")
                path = dataset_info.get("path")
                if isinstance(path, pathlib.Path):
                    st.info(f"資料夾: {path}")
                else:
                    st.info("未指定資料夾")

        has_documents = bool(dataset_info.get("has_documents"))
        
        with col2:
            st.subheader("⚙️ 分析設定")
            limit_files = st.number_input("限制處理檔案數量 (0=全部)", min_value=0, value=3)
            if limit_files == 0:
                limit_files = None
            
            use_llm_validation = st.checkbox("使用LLM驗證", value=True)
            include_json_search = st.checkbox("啟用主題字補查", value=False, 
                                             help="從 mojLawSplitJSON 補查未找到的主題字")
            
            if st.button("🚀 開始完整分析", type="primary", use_container_width=True):
                if not has_documents:
                    st.error("請提供至少一個法規 TXT 檔案")
                else:
                    st.info("🔄 開始完整法規分析...")
                    
                    rows = analysis_engine.analyze_full(
                        dataset_source,
                        keyword_source,
                        limit_files,
                        use_llm_validation,
                        include_json_search,
                        selected_model,
                    )
                    
                    df = StreamlitUI.display_results(rows)
                    
                    if df is not None:
                        st.subheader("📥 下載結果")
                        col_download1, col_download2 = st.columns(2)
                        
                        excel_filename, csv_filename = DownloadHandler.build_export_filenames("lawhits", selected_model)

                        with col_download1:
                            excel_buf = DownloadHandler.to_downloadable_excel(df, excel_filename)
                            if excel_buf:
                                st.download_button(
                                    label="📊 下載 Excel 檔案",
                                    data=excel_buf,
                                    file_name=excel_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )

                        with col_download2:
                            csv_data = DownloadHandler.to_downloadable_csv(df, csv_filename)
                            if csv_data:
                                st.download_button(
                                    label="📋 下載 CSV 檔案",
                                    data=csv_data,
                                    file_name=csv_filename,
                                    mime="text/csv",
                                    use_container_width=True
                                )
    
    else:
        # AI 聊天模式
        st.header("💬 AI 聊天對話")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("請輸入您的問題..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    if ai_service.provider == AIProvider.GEMINI:
                        # Gemini 串流處理
                        if use_stream:
                            response = ai_service.chat_completion(
                                st.session_state.messages,
                                selected_model,
                                stream=True
                            )
                            
                            if response:
                                for chunk in response:
                                    if hasattr(chunk, 'text'):
                                        full_response += chunk.text
                                        message_placeholder.markdown(full_response + "▌")
                        else:
                            response = ai_service.chat_completion(
                                st.session_state.messages,
                                selected_model,
                                stream=False
                            )
                            if response and hasattr(response, 'text'):
                                full_response = response.text
                                message_placeholder.markdown(full_response)
                    else:
                        # OpenAI 或 Ollama
                        extra_params = {}
                        if ai_service.provider == AIProvider.OLLAMA:
                            extra_params["extra_body"] = {"options": llm_processor.ollama_options}
                        
                        if use_stream:
                            # 使用串流模式
                            try:
                                response = ai_service.chat_completion(
                                    st.session_state.messages,
                                    selected_model,
                                    stream=True,
                                    **extra_params
                                )
                                
                                if response:
                                    for chunk in response:
                                        if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content is not None:
                                            full_response += chunk.choices[0].delta.content
                                            message_placeholder.markdown(full_response + "▌")
                            
                            except Exception as stream_error:
                                # 如果串流失敗（如組織未驗證），自動降級為非串流模式
                                if "stream" in str(stream_error).lower() or "unsupported_value" in str(stream_error).lower():
                                    message_placeholder.markdown("⚠️ 串流模式不可用，自動切換為標準模式...\n\n")
                                    response = ai_service.chat_completion(
                                        st.session_state.messages,
                                        selected_model,
                                        stream=False,
                                        **extra_params
                                    )
                                    if response and hasattr(response, 'choices') and response.choices:
                                        full_response = response.choices[0].message.content or ""
                                        message_placeholder.markdown(full_response)
                                else:
                                    raise stream_error
                        else:
                            # 使用非串流模式
                            response = ai_service.chat_completion(
                                st.session_state.messages,
                                selected_model,
                                stream=False,
                                **extra_params
                            )
                            if response and hasattr(response, 'choices') and response.choices:
                                full_response = response.choices[0].message.content or ""
                                message_placeholder.markdown(full_response)
                    
                    if full_response:
                        message_placeholder.markdown(full_response)
                
                except Exception as e:
                    error_msg = f"❌ 錯誤: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    full_response = error_msg
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
