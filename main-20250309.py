from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# 1. 状態クラスの定義
class KnowledgeGap(BaseModel):
    topic: str = Field(description="知識ギャップのトピック")
    question: str = Field(description="そのギャップに関する質問")
    importance: int = Field(description="重要度（1-10）")

class ProgressiveRAGState(BaseModel):
    # 入力
    question: str = Field(description="ユーザーの元の質問")
    
    # 状態追跡
    current_depth: int = Field(default=0, description="現在の調査深度")
    max_depth: int = Field(default=3, description="最大調査深度")
    accumulated_knowledge: Dict[str, Any] = Field(default_factory=dict, description="蓄積された知識")
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list, description="特定された知識ギャップ")
    focus_questions: List[str] = Field(default_factory=list, description="次の調査の焦点となる質問")
    
    # 出力
    final_answer: Optional[str] = Field(default=None, description="最終的な回答")

# 2. 各ノードの処理関数
def perform_initial_retrieval(state: ProgressiveRAGState) -> ProgressiveRAGState:
    """初期的な情報検索を行う"""
    print(f"初期検索: {state.question}")
    
    # 実際の実装ではここでRAGシステムを用いて関連文書を検索
    # この例ではLLM単体で代用
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 初期検索テンプレート
    initial_retrieval_template = ChatPromptTemplate.from_template(
        """
        以下の質問に対する基本的な情報を提供してください:
        {question}
        
        できるだけ事実に基づいた簡潔な回答を提供してください。
        """
    )
    
    # LLMで初期回答を生成
    chain = initial_retrieval_template | llm
    initial_knowledge = chain.invoke({"question": state.question})
    
    # 状態を更新
    state.accumulated_knowledge["initial_info"] = initial_knowledge.content
    
    return state

def analyze_current_knowledge(state: ProgressiveRAGState) -> ProgressiveRAGState:
    """現在の知識を分析し、ギャップを特定する"""
    print(f"知識分析 (深度: {state.current_depth})")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 知識分析テンプレート
    knowledge_analysis_template = ChatPromptTemplate.from_template(
        """
        以下の質問に答えるための知識ギャップを分析してください:
        {question}
        
        現在の知識:
        {accumulated_knowledge}
        
        以下のJSON形式で3つの主要な知識ギャップを特定してください:
        ```json
        [
            {{
                "topic": "ギャップのトピック",
                "question": "そのギャップに関する具体的な質問",
                "importance": 重要度（1-10）
            }},
            ...
        ]
        ```
        
        現在の調査深度: {current_depth}
        最大調査深度: {max_depth}
        
        深度が深くなるほど、より詳細で専門的なギャップを特定してください。
        """
    )
    
    # 知識のテキスト形式への変換
    knowledge_text = "\n\n".join([f"{k}: {v}" for k, v in state.accumulated_knowledge.items()])
    
    # LLMで知識ギャップを分析
    chain = knowledge_analysis_template | llm
    analysis_result = chain.invoke({
        "question": state.question,
        "accumulated_knowledge": knowledge_text,
        "current_depth": state.current_depth,
        "max_depth": state.max_depth
    })
    
    # 結果の解析
    try:
        # JSONを抽出
        json_str = analysis_result.content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        
        # 知識ギャップのパース
        gaps = json.loads(json_str)
        state.knowledge_gaps = [KnowledgeGap(**gap) for gap in gaps]
        
        # 焦点質問の作成
        state.focus_questions = [gap.question for gap in state.knowledge_gaps]
        
    except Exception as e:
        print(f"知識ギャップの解析エラー: {e}")
        # エラー時は空のギャップを設定
        state.knowledge_gaps = []
        state.focus_questions = []
    
    return state

def retrieve_focused_information(state: ProgressiveRAGState) -> ProgressiveRAGState:
    """特定された知識ギャップに焦点を当てた情報検索"""
    print(f"焦点検索 (深度: {state.current_depth}, 質問数: {len(state.focus_questions)})")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 焦点検索テンプレート
    focused_retrieval_template = ChatPromptTemplate.from_template(
        """
        以下の質問に詳細に回答してください:
        {focus_question}
        
        この質問は以下のより大きな質問の一部です:
        {original_question}
        
        現在の調査深度: {current_depth}
        最大調査深度: {max_depth}
        
        調査深度に応じて、より詳細で専門的な情報を提供してください。
        できるだけ事実に基づいた、具体的かつ詳細な回答を提供してください。
        """
    )
    
    # 各焦点質問に対して検索を実行
    for i, focus_question in enumerate(state.focus_questions):
        chain = focused_retrieval_template | llm
        focus_result = chain.invoke({
            "focus_question": focus_question,
            "original_question": state.question,
            "current_depth": state.current_depth,
            "max_depth": state.max_depth
        })
        
        # 結果を蓄積された知識に追加
        key = f"depth_{state.current_depth}_focus_{i}"
        state.accumulated_knowledge[key] = focus_result.content
    
    return state

def integrate_new_knowledge(state: ProgressiveRAGState) -> ProgressiveRAGState:
    """新しく取得した知識を統合する"""
    print(f"知識統合 (深度: {state.current_depth})")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 知識統合テンプレート
    knowledge_integration_template = ChatPromptTemplate.from_template(
        """
        以下の元の質問に関連する知識を統合してください:
        {question}
        
        以下の知識ソースを統合し、一貫性のある包括的な理解を形成してください:
        {knowledge_text}
        
        現在の調査深度: {current_depth}
        
        重要な概念、関連性、パターンを特定してください。矛盾がある場合は指摘してください。
        1000字以内の簡潔な統合された知識要約を提供してください。
        """
    )
    
    # 最新の知識だけを選択（現在の深度に関連するもの）
    current_depth_keys = [k for k in state.accumulated_knowledge.keys() if f"depth_{state.current_depth}_" in k]
    recent_knowledge = {k: state.accumulated_knowledge[k] for k in current_depth_keys}
    
    # 知識のテキスト形式への変換
    knowledge_text = "\n\n".join([f"{k}: {v}" for k, v in recent_knowledge.items()])
    
    # LLMで知識統合を実行
    chain = knowledge_integration_template | llm
    integration_result = chain.invoke({
        "question": state.question,
        "knowledge_text": knowledge_text,
        "current_depth": state.current_depth
    })
    
    # 統合結果を蓄積された知識に追加
    state.accumulated_knowledge[f"integration_depth_{state.current_depth}"] = integration_result.content
    
    return state

def generate_final_answer(state: ProgressiveRAGState) -> ProgressiveRAGState:
    """蓄積された知識から最終回答を生成"""
    print("最終回答生成")
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 最終回答テンプレート
    final_answer_template = ChatPromptTemplate.from_template(
        """
        以下の質問に対する総合的な回答を提供してください:
        {question}
        
        以下は調査プロセスで蓄積された知識です:
        {accumulated_knowledge}
        
        上記の知識に基づいて、質問に対する包括的かつ事実に基づいた回答を作成してください。
        情報が不十分な場合は、その点を正直に示してください。
        
        回答は明確に構造化し、最も重要な点から始めてください。
        """
    )
    
    # 統合された知識だけを選択
    integration_keys = [k for k in state.accumulated_knowledge.keys() if "integration" in k]
    integrated_knowledge = {k: state.accumulated_knowledge[k] for k in integration_keys}
    
    # 初期情報も含める
    if "initial_info" in state.accumulated_knowledge:
        integrated_knowledge["initial_info"] = state.accumulated_knowledge["initial_info"]
    
    # 知識のテキスト形式への変換
    knowledge_text = "\n\n".join([f"{k}: {v}" for k, v in integrated_knowledge.items()])
    
    # LLMで最終回答を生成
    chain = final_answer_template | llm
    final_result = chain.invoke({
        "question": state.question,
        "accumulated_knowledge": knowledge_text
    })
    
    # 最終回答を設定
    state.final_answer = final_result.content
    
    return state

# 3. 条件関数：調査を深めるかどうか
def determine_investigation_depth(state: ProgressiveRAGState) -> str:
    """現在の知識状態から次の調査ステップを決定"""
    
    # 知識ギャップがなくなった、または最大深度に達した場合は終了
    if (not state.knowledge_gaps or state.current_depth >= state.max_depth):
        return "finalize"
    else:
        # 深度を増加
        state.current_depth += 1
        return "deepen_investigation"

# 4. グラフの構築
def build_progressive_rag_graph():
    """進行的RAGグラフを構築"""
    
    # グラフの作成
    builder = StateGraph(ProgressiveRAGState)
    
    # ノードの追加
    builder.add_node("initial_retrieval", perform_initial_retrieval)
    builder.add_node("knowledge_analyzer", analyze_current_knowledge)
    builder.add_node("focused_retrieval", retrieve_focused_information)
    builder.add_node("knowledge_integrator", integrate_new_knowledge)
    builder.add_node("final_answer_generator", generate_final_answer)
    
    # エッジの追加
    builder.add_edge("initial_retrieval", "knowledge_analyzer")
    
    # 条件分岐: 知識ギャップに基づく
    builder.add_conditional_edges(
        "knowledge_analyzer",
        determine_investigation_depth,
        {
            "deepen_investigation": "focused_retrieval",
            "finalize": "final_answer_generator"
        }
    )
    
    builder.add_edge("focused_retrieval", "knowledge_integrator")
    builder.add_edge("knowledge_integrator", "knowledge_analyzer")
    builder.add_edge("final_answer_generator", END)
    
    # グラフのコンパイル
    return builder.compile()

# 5. 実行関数
def run_progressive_rag(question: str, max_depth: int = 3):
    """進行的RAGを実行"""
    
    # グラフの構築
    graph = build_progressive_rag_graph()
    
    # 初期状態の設定
    initial_state = ProgressiveRAGState(
        question=question,
        max_depth=max_depth
    )
    
    # グラフの実行
    result = graph.invoke(initial_state)
    
    return result

# 使用例
if __name__ == "__main__":
    question = "量子コンピュータが現代の暗号技術に与える影響とその対策について詳細に説明してください。"
    result = run_progressive_rag(question, max_depth=3)
    print("\n最終回答:")
    print(result.final_answer)
