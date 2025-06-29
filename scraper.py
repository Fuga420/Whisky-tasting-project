import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from pathlib import Path

# --- プログラムの本体 ---

# 1. URLリストが書かれたファイル名を指定
url_file = 'whisky_urls.txt' 

# 2. 抽出した全データを格納するための空のリストを用意
all_whisky_data = []

# 3. URLリストを読み込む
try:
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"--- {len(urls)}件のURLを読み込みました。詳細情報の収集を開始します。 ---")
except FileNotFoundError:
    print(f"エラー: '{url_file}' が見つかりません。先に`url_collector.py`を実行してください。")
    exit()

# 4. 各URLを順番に処理
for i, url in enumerate(urls):
    print(f"\n--- {i+1}/{len(urls)}件目を処理中: {url} ---")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # ▼ 商品名の取得
        product_name_tag = soup.find('h1', class_='ty-product-block-title')
        product_name = product_name_tag.find('bdi').text.strip() if product_name_tag else '商品名不明'

        # ▼ テイスティングノートの取得
        description_area = soup.find('div', id='content_description')
        tasting_note_parts = []
        if description_area:
            paragraphs = description_area.find_all('p')
            for p in paragraphs:
                text = p.text.strip()
                if text.startswith(('香り：', '味わい：', 'フィニッシュ：', 'アロマ：', 'フレーバー：')):
                    tasting_note_parts.append(text)
        tasting_note = '\n'.join(tasting_note_parts)

        # ▼ スペック情報の取得
        spec_data = {}
        spec_area = soup.find('div', id='content_features')
        if spec_area:
            spec_items = spec_area.find_all('div', class_='ty-product-feature')
            for item in spec_items:
                key_tag = item.find('span', class_='ty-product-feature__label')
                value_tag = item.find('div', class_='ty-product-feature__value')
                if key_tag and value_tag:
                    key = key_tag.text.strip().replace(':', '')
                    value = value_tag.text.strip()
                    spec_data[key] = value

        # ★★★【最新の修正】高画質な画像URLを取得するロジック ★★★
        image_link_tag = soup.find('a', class_='cm-image-previewer')
        image_url = image_link_tag['href'] if image_link_tag and image_link_tag.has_attr('href') else 'URLなし'
        
        # １ページ分の全データを一つの辞書にまとめる
        single_whisky_data = {
            '商品名': product_name,
            'テイスティングノート': tasting_note,
            'URL': url,
            '画像URL': image_url, # 抽出した画像URLを追加
            **spec_data
        }
        
        # データをメモリ上のリストに溜め込む（カラムずれ対策）
        all_whisky_data.append(single_whisky_data)
        print(f"○ 「{product_name}」のデータを正常に取得しました。")

    except Exception as e:
        print(f"× エラーが発生したため、このURLの処理をスキップします: {e}")
    
    # サーバーに配慮して待機
    wait_time = random.uniform(2, 4)
    print(f"   {wait_time:.2f}秒待機します...")
    time.sleep(wait_time)

# 5. 【ループ後】全データが溜まったリストを、一括でCSVファイルに保存
if all_whisky_data:
    # Pandasがカラムの有無を自動で判断し、空欄（NaN）で埋めてくれる
    df = pd.DataFrame(all_whisky_data)
    
    # 保存先フォルダとファイルパスを指定
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / 'whisky_dataset_final.csv'
    
    # CSVファイルとして出力
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    print(f"\n--- 全ての処理が完了しました！ ---")
    print(f"{len(all_whisky_data)}件のデータを抽出し、'{file_path}'に保存しました。")
else:
    print("\nデータを一件も抽出できませんでした。")