import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer # ← MeCabの代わりにJanomeをインポート

# --- データとモデルの読み込み ---
@st.cache_data
def load_data_and_model():
    print("--- データの読み込みとモデルの構築を開始します（このメッセージは初回のみ表示されます） ---")
    
    # 1. データの読み込みと前準備
    csv_file_path = Path('output') / 'whisky_dataset_final.csv'
    df = pd.read_csv(csv_file_path)
    df.dropna(subset=['テイスティングノート'], inplace=True)
    df = df[df['テイスティングノート'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    # 2. 形態素解析とクリーニング（Janomeを使用）
    # ▼▼▼ MeCabからJanomeへの変更箇所 ▼▼▼
    t = Tokenizer() # JanomeのTokenizerを準備
    stop_words = ['する', 'いる', 'ある', 'これ', 'それ', 'あれ', '思う', '感じ', '的',
                  '香り', '味わい', 'フィニッシュ', 'アロマ', 'フレーバー', 'ノート', 'テイスティング']

    def clean_and_tokenize(text):
        # Janomeで単語に分割
        words = [token.surface for token in t.tokenize(text)]
        
        # フィルタリング処理
        cleaned_words = [word for word in words if word not in stop_words and len(word) > 1 and not word.isnumeric()]
        return ' '.join(cleaned_words)
    # ▲▲▲ MeCabからJanomeへの変更箇所 ▲▲▲

    df['tokens_cleaned'] = df['テイスティングノート'].apply(clean_and_tokenize)
    
    # 3. キーワードの重みづけ（この部分は変更なし）
    smoky_words = ['スモーキー', 'ピート', 'ピーティ', 'ヨード', '煙', '燻製', '潮', '薬品', '正露丸']
    fruity_words = ['フルーティー', 'フルーツ', '果実', 'リンゴ', '柑橘', 'レモン', 'オレンジ', 'ベリー', 'レーズン', 'ピーチ', 'アプリコット', 'パイナップル']
    sherry_words = ['シェリー', 'ドライフルーツ', 'レーズン', 'カカオ', 'チョコレート']

    def add_weights(text):
        if not isinstance(text, str): return ""
        boosted_text = text
        if any(word in text for word in smoky_words): boosted_text += ' スモーキー' * 10
        if any(word in text for word in fruity_words): boosted_text += ' フルーティー' * 1.5
        if any(word in text for word in sherry_words): boosted_text += ' シェリー' * 1.5
        return boosted_text
    
    df['tokens_boosted'] = df['tokens_cleaned'].apply(add_weights)
    
    # 4. TF-IDFと類似度計算（この部分は変更なし）
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(df['tokens_boosted'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    
    indices = pd.Series(df.index, index=df['商品名']).drop_duplicates()

    print("--- モデルの準備が完了しました ---")
    
    return df, cosine_sim_matrix, indices

# --- アプリケーションのUI部分（ここは変更なし） ---
st.title('ウイスキーおすすめ')
st.write('あなたの好きなウイスキーを選ぶと、AIが風味の似たおすすめのウイスキーを提案します。')

df, cosine_sim, indices = load_data_and_model()

whisky_list = df['商品名'].tolist()
selected_whisky = st.selectbox('好きなウイスキーを1本選んでください', whisky_list)

if st.button('このウイスキーに似たお酒を探す'):
    if selected_whisky:
        try:
            idx = indices[selected_whisky]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
            whisky_indices = [i[0] for i in sim_scores]
            
            recommendations = df.iloc[whisky_indices]

            st.subheader(f'「{selected_whisky}」が好きなあなたへのおすすめはこちらです！')
            for index, row in recommendations.iterrows():
                st.markdown(f"#### {row['商品名']}")
                st.markdown(f"**テイスティングノート:** {row['テイスティングノート']}")
                st.markdown(f"[商品ページへ]({row['URL']})")
                st.markdown("---")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")