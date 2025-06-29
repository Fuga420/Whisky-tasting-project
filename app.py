import streamlit as st
import pandas as pd
import random 
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer

# --- データとモデルの読み込み（この関数は変更なし） ---
@st.cache_data
def load_data_and_model():
    print("--- データの読み込みとモデルの構築を開始します ---")
    
    csv_file_path = Path('output') / 'whisky_dataset_final.csv'
    df = pd.read_csv(csv_file_path)
    
    df['年数'] = pd.to_numeric(df['年数'], errors='coerce')
    df.dropna(subset=['商品名', 'テイスティングノート'], inplace=True)
    df = df[df['テイスティングノート'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    t = Tokenizer()
    stop_words = ['する', 'いる', 'ある', 'これ', 'それ', 'あれ', '思う', '感じ', '的',
                  '香り', '味わい', 'フィニッシュ', 'アロマ', 'フレーバー', 'ノート', 'テイスティング']
    def clean_and_tokenize(text):
        words = [token.surface for token in t.tokenize(text)]
        cleaned_words = [word for word in words if word not in stop_words and len(word) > 1 and not word.isnumeric()]
        return ' '.join(cleaned_words)
    df['tokens_cleaned'] = df['テイスティングノート'].apply(clean_and_tokenize)
    
    smoky_words = ['スモーキー', 'ピート', 'ピーティ', 'ヨード', '煙', '燻製', '潮', '薬品', '正露丸']
    fruity_words = ['フルーティー', 'フルーツ', '果実', 'リンゴ', '柑橘', 'レモン', 'オレンジ', 'ベリー', 'レーズン', 'ピーチ', 'アプリコット', 'パイナップル']
    sherry_words = ['シェリー', 'ドライフルーツ', 'レーズン', 'カカオ', 'チョコレート']
    def add_weights(text):
        if not isinstance(text, str): return ""
        boosted_text = text
        if any(word in text for word in smoky_words): boosted_text += ' スモーキー' * 10
        if any(word in text for word in fruity_words): boosted_text += ' フルーティー' * 5
        if any(word in text for word in sherry_words): boosted_text += ' シェリー' * 8
        return boosted_text
    df['tokens_boosted'] = df['tokens_cleaned'].apply(add_weights)
    
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(df['tokens_boosted'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    indices = pd.Series(df.index, index=df['商品名']).drop_duplicates()
    
    print("--- モデルの準備が完了しました ---")
    return df, cosine_sim_matrix, indices, vectorizer, tfidf_matrix

# --- レコメンド関数（変更なし） ---
def get_recommendations_with_serendipity(title, df, cosine_sim, indices):
    try:
        idx = indices[title]
    except KeyError: return pd.DataFrame()
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_candidates_indices = [i[0] for i in sim_scores[1:31]]
    if len(top_candidates_indices) < 5: return df.iloc[top_candidates_indices]
    final_recommendations_indices = top_candidates_indices[:2]
    remaining_candidates = top_candidates_indices[2:]
    random_sample_indices = random.sample(remaining_candidates, 3)
    final_recommendations_indices.extend(random_sample_indices)
    return df.iloc[final_recommendations_indices]

def recommend_by_flavor(keywords, vectorizer, tfidf_matrix, df, top_n=5):
    keyword_vec = vectorizer.transform(keywords)
    cosine_sim = cosine_similarity(keyword_vec, tfidf_matrix)
    top_indices = cosine_sim.flatten().argsort()[::-1][:top_n]
    return df.iloc[top_indices]

# --- アプリケーションのUI（見た目）部分 ---
st.set_page_config(layout="wide")
st.title('AIウイスキーソムリエ 🥃')
st.write('あなたの好みから、次の一本をお探しします。')

with st.spinner('AIソムリエが厨房で準備をしています... 初回起動には数分かかります。'):
    df, cosine_sim, indices, vectorizer, tfidf_matrix = load_data_and_model()

st.success('準備が完了しました！いつでもご注文をどうぞ。')

# --- ▼▼▼ サイドバー（絞り込みオプション）の修正 ▼▼▼ ---
st.sidebar.header('詳細検索オプション')

# 熟成年数のスライダー
min_age_in_data = int(df['年数'].dropna().min())
max_age_in_data = int(df['年数'].dropna().max())
selected_age = st.sidebar.slider(
    '熟成年数の範囲',
    min_value=min_age_in_data,
    max_value=max_age_in_data,
    value=(min_age_in_data, max_age_in_data)
)

# 地域のマルチセレクト（指定されたリストに変更）
region_list = ['全地域', 'アイラ', 'キャンベルタウン', 'スペイサイド', 'アイランズ', 'ハイランド', 'ローランド']
selected_regions = st.sidebar.multiselect(
    '生産地域（複数選択可）',
    region_list
)

# ブランドのマルチセレクトはここから削除しました

# --- メイン画面（検索機能） ---
st.markdown("---")
left_column, right_column = st.columns(2)

with left_column:
    st.header('好きな銘柄から探す')
    whisky_list = df['商品名'].tolist()[::-1]
    selected_whisky = st.selectbox('好きなウイスキーを1本選んでください', whisky_list, key='item_select')
    if st.button('このウイスキーに似たお酒を探す', key='item_button'):
        recommendations = get_recommendations_with_serendipity(selected_whisky, df, cosine_sim, indices)
        st.session_state['recommendations'] = recommendations
        st.session_state['search_title'] = f'「{selected_whisky}」が好きなあなたへのおすすめ'

with right_column:
    st.header('好きな香味から探す')
    flavor_input = st.text_input('好きな香味やキーワードを入力してください', placeholder='例：フルーティー, 甘い, 少しスモーキー', key='flavor_input')
    if st.button('この香味に近いお酒を探す', key='flavor_button'):
        if flavor_input:
            recommendations = recommend_by_flavor([flavor_input], vectorizer, tfidf_matrix, df)
            st.session_state['recommendations'] = recommendations
            st.session_state['search_title'] = f'「{flavor_input}」のイメージに近いおすすめ'

# --- 結果表示エリア ---
if 'recommendations' in st.session_state:
    st.markdown("---")
    st.header('AIからのご提案')
    
    recommendations_df = st.session_state['recommendations']
    filtered_results = recommendations_df.copy()

    # フィルタリング処理
    filtered_results.dropna(subset=['年数'], inplace=True)
    filtered_results = filtered_results[(filtered_results['年数'] >= selected_age[0]) & (filtered_results['年数'] <= selected_age[1])]
    if selected_regions and '全地域' not in selected_regions:
        filtered_results = filtered_results[filtered_results['地域'].isin(selected_regions)]
    
    # ブランドによるフィルタリング処理をここから削除しました

    st.subheader(st.session_state['search_title'])
    
    if filtered_results.empty:
        st.warning('条件に合うウイスキーが見つかりませんでした。絞り込み条件を緩めてみてください。')
    else:
        for index, row in filtered_results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if '画像URL' in row and pd.notna(row['画像URL']) and row['画像URL'] != 'URLなし':
                        st.image(row['画像URL'], width=150)
                with col2:
                    st.markdown(f"#### {row['商品名']}")
                    if 'ブランド' in row and pd.notna(row['ブランド']):
                        st.markdown(f"**ブランド:** {row.get('ブランド', '情報なし')} | **地域:** {row.get('地域', '情報なし')}")
                    with st.expander("テイスティングノートを見る"):
                        st.write(row.get('テイスティングノート', '情報なし'))
                    st.markdown(f"[商品ページへ]({row.get('URL', '#')})")