import streamlit as st
import torch
import torch.nn as nn
import json
import re
import time

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="HDFS Log Anomaly Detector",
    page_icon="🛡️",
    layout="wide"
)

# Пути к файлам
MODEL_PATH = 'lstm_model_weights.pt'
VOCAB_PATH = 'event_to_int_vocab.json'
LOG_FILE_PATH = 'HDFS.log'


# ==========================================
# 2. ОПРЕДЕЛЕНИЕ МОДЕЛИ (ОБНОВЛЕННОЕ ПОД НОВУЮ ТРЕНИРОВКУ)
# ==========================================
class LSTMClassifier(nn.Module):
    # Добавили параметр dropout=0.5, как при обучении
    def __init__(self, vocab_size, emb_dim=64, hid_dim=128, out_dim=2, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Важно: при обучении был dropout, тут он тоже должен быть в определении,
        # хотя в режиме eval() он отключится сам.
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Берем последний выход
        out = hidden[-1]
        out = self.dropout(out)  # Слой есть, но в eval() он просто пропустит данные
        return self.fc(out)


# ==========================================
# 3. УЛУЧШЕННЫЙ ПАРСЕР
# ==========================================
class HDFSLogParser:
    def __init__(self):
        self.blk_pattern = re.compile(r"(blk_[-0-9]+)")
        self.signatures = {
            "Receiving block": "E2",
            "Received block": "E22",
            "PacketResponder": "E5",
            "Served block": "E3",
            "verification succeeded": "E26",
            "addStoredBlock": "E11",
            "allocateBlock": "E9",
            "Deleting block": "E25",
            "ask": "E27",
            "Exception": "E_Error",
            "warn": "E_Warn"
        }

    def parse_line(self, line):
        match = self.blk_pattern.search(line)
        if not match: return None, None
        block_id = match.group(1)
        event_type = "Unknown"
        for key, eid in self.signatures.items():
            if key in line:
                event_type = eid
                break
        return block_id, event_type


# ==========================================
# 4. ФУНКЦИИ ЗАГРУЗКИ
# ==========================================
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open(VOCAB_PATH, 'r') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        st.error(f"Файл {VOCAB_PATH} не найден!")
        st.stop()

    vocab_size = len(vocab) + 1

    # === ИСПРАВЛЕНИЕ ЗДЕСЬ ===
    # Было: 100, Стало: 64 (так как новая модель училась с emb_dim=64)
    # Параметры: vocab_size, emb_dim=64, hid_dim=128, out_dim=2, n_layers=2
    model = LSTMClassifier(vocab_size, emb_dim=64, hid_dim=128, out_dim=2, n_layers=2).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # Переключаем в режим оценки (выключает Dropout)
    except FileNotFoundError:
        st.error(f"Файл весов {MODEL_PATH} не найден!")
        st.stop()
    except RuntimeError as e:
        st.error(f"Несовпадение размеров модели! Удалите старый .pt файл и скачайте новый. Ошибка: {e}")
        st.stop()

    return model, vocab, device


# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
model, vocab, device = load_resources()
parser = HDFSLogParser()

st.sidebar.title("⚙️ Панель управления")
st.sidebar.success(f"Модель загружена. Device: {device}")
speed = st.sidebar.slider("Скорость чтения логов (сек)", 0.01, 1.0, 0.1)

st.title("🛡️ AI Log Sentinel")
st.markdown("Система обнаружения аномалий в логах HDFS на базе **LSTM**.")

tab1, tab2 = st.tabs(["📡 Мониторинг потока (Live)", "🔍 Ручная проверка"])

# --- ПОТОК ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Поток данных из HDFS.log")
    with col2:
        start_btn = st.button("▶ ЗАПУСТИТЬ ПОТОК")

    m1, m2 = st.columns(2)
    metric_ok = m1.empty()
    metric_anom = m2.empty()
    log_container = st.container(height=300, border=True)

    if 'count_ok' not in st.session_state: st.session_state.count_ok = 0
    if 'count_anom' not in st.session_state: st.session_state.count_anom = 0

    if start_btn:
        active_sessions = {}
        try:
            with open(LOG_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    blk_id, event_str = parser.parse_line(line)
                    if not blk_id or event_str == "Unknown": continue

                    event_idx = vocab.get(event_str, 0)
                    if blk_id not in active_sessions: active_sessions[blk_id] = []
                    active_sessions[blk_id].append(event_idx)

                    # --- НОВАЯ ЛОГИКА ТРИГГЕРА (БЕЗ E5) ---
                    should_predict = False
                    if event_str in ["E26", "E25"]:
                        should_predict = True
                    elif len(active_sessions[blk_id]) > 40:
                        should_predict = True

                    if should_predict:
                        sequence = active_sessions[blk_id]
                        # Игнорируем слишком короткие обрывки, если это не явная верификация
                        if len(sequence) < 3 and event_str != "E26":
                            del active_sessions[blk_id]
                            continue

                        tensor = torch.tensor([sequence], dtype=torch.long).to(device)

                        # ПРЕДСКАЗАНИЕ (ЧИСТОЕ, БЕЗ КОСТЫЛЕЙ)
                        is_anomaly = False
                        with torch.no_grad():
                            out = model(tensor)
                            probs = torch.softmax(out, dim=1)
                            confidence = probs[0][1].item()
                            _, pred = torch.max(out, 1)
                            if pred.item() == 1: is_anomaly = True

                        if is_anomaly:
                            st.session_state.count_anom += 1
                            log_container.error(
                                f"🚨 АНОМАЛИЯ! Block: {blk_id} | Len: {len(sequence)} | Conf: {confidence:.2%}")
                        else:
                            st.session_state.count_ok += 1
                            msg = "✅ Verified" if event_str == "E26" else "ℹ️ Ends"
                            log_container.success(f"{msg}: {blk_id} | Len: {len(sequence)}")

                        del active_sessions[blk_id]
                        metric_ok.metric("Норма", st.session_state.count_ok)
                        metric_anom.metric("Аномалии", st.session_state.count_anom)
                        time.sleep(speed)

        except FileNotFoundError:
            st.error("Файл HDFS.log не найден!")

# --- РУЧНОЙ ВВОД ---
with tab2:
    st.header("Ручной анализ")
    st.info("Введите коды событий через пробел.")

    # Дефолтное значение - норма
    user_input = st.text_input("События:", "E5 E22 E11 E9 E11 E9 E26 E26 E26")

    # Ползунок чувствительности (фича для защиты!)
    threshold = st.slider("Порог срабатывания тревоги (%)", 1, 100, 15) / 100.0

    if st.button("Проверить"):
        tokens = user_input.strip().split()
        numeric_seq = [vocab[t] for t in tokens if t in vocab]

        if not numeric_seq:
            st.error("Нет известных событий")
        else:
            tensor = torch.tensor([numeric_seq], dtype=torch.long).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                prob_anom = probs[0][1].item()  # Вероятность аномалии (0.0 - 1.0)

            # === ЛОГИКА ПРИНЯТИЯ РЕШЕНИЯ ===
            is_anomaly = False
            reason = ""

            # 1. Проверка по Нейросети (с учетом порога)
            if prob_anom > threshold:
                is_anomaly = True
                # reason = f"Нейросеть обнаружила подозрительный паттерн (Risk > {int(threshold * 100)}%)"

            # 2. Эвристическая проверка (Защита от E5 E5 E5...)
            # Если в цепочке только 1 или 2 уникальных события, но длина большая - это спам/DOS
            # unique_events = len(set(numeric_seq))
            # if len(numeric_seq) > 5 and unique_events < 2:
            #     is_anomaly = True
            #     prob_anom = 0.99  # Принудительно повышаем риск для UI
                # reason = "Обнаружено зацикливание событий (DoS паттерн)"

            # === ВЫВОД ===
            st.metric("Вероятность аномалии (AI)", f"{prob_anom:.2%}")

            if is_anomaly:
                st.error(f"РЕЗУЛЬТАТ: 🛑 ОБНАРУЖЕНА УГРОЗА")
                st.warning(f"Причина: {reason}")
            else:
                st.success("РЕЗУЛЬТАТ: ✅ ВСЁ ЧИСТО")