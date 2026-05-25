import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool
import joblib

def train_and_save_model():
    print("🚀 Старт процесса обучения модели...")
    
    # 1. Загрузка данных
    data_path = "../data/Main_dataset.xlsx" 
    if not os.path.exists(data_path):
        data_path = "data/Main_dataset.xlsx"
        
    if not os.path.exists(data_path):
        data_path = "Main_dataset.xlsx - Sheet1.csv"
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
        
    print(f"📊 Исходный датасет загружен. Всего строк: {len(df)}")

    # ==================== БЛОК ОЧИСТКИ ОТ ВЫБРОСОВ И АНОМАЛИЙ ====================
    print("🧹 Запуск процесса очистки данных...")
    target_column = 'Цена'
    
    # Удаляем строки без цены или с некорректной ценой
    df = df.dropna(subset=[target_column])
    df = df[df[target_column] > 0]
    
    # Фильтр 1: Логическая проверка площадей
    df = df[
        (df['Жилая площадь'].fillna(0) < df['Общая площадь']) & 
        (df['Площадь кухни'].fillna(0) < df['Общая площадь'])
    ]
    
    # [Внедрение пункта 8]: Очистка по стоимости кв.м. методом IQR (Межквартильный размах)
    df['temp_price_m2'] = df[target_column] / df['Общая площадь']
    
    q1 = df['temp_price_m2'].quantile(0.25)
    q3 = df['temp_price_m2'].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Страховка, чтобы нижняя граница не ушла ниже нуля
    if lower_bound < 30000: 
        lower_bound = df['temp_price_m2'].quantile(0.01)
        
    df = df[(df['temp_price_m2'] >= lower_bound) & (df['temp_price_m2'] <= upper_bound)]
    
    print(f"   [Инфо]: Границы цен определены методом IQR")
    print(f"   [Инфо]: Минимальная отсеченная цена метра: {lower_bound:,.2f} ₽".replace(",", " "))
    print(f"   [Инфо]: Максимальная отсеченная цена метра: {upper_bound:,.2f} ₽".replace(",", " "))
    
    df = df.drop(columns=['temp_price_m2'])
    print(f"✅ Очистка завершена. Осталось строк для обучения: {len(df)}")
    # ============================================================================

    # ==================== FEATURE ENGINEERING ====================
    print("🛠 Проектирование новых признаков (Feature Engineering)...")
    


    # [Внедрение пункта 2]: Расчет относительных площадей
    df['Доля кухни'] = df['Площадь кухни'] / df['Общая площадь']
    df['Доля жилой площади'] = df['Жилая площадь'] / df['Общая площадь']
    
    # Списки признаков
    numerical_features = [
        'Общая площадь',           
        'Доля кухни',     # Новый
        'Доля жилой площади',      # Новый
        'Высота потолков', 
        'Возраст дома',        
        'Этаж', 
        'Этажность дома', 
        'Расстояние до центра (м)',
        'Расстояние до вокзала Краснодар-1 (м)', 
        'Расстояние до аэропорта (м)',
        'Расстояние до парка (м)',
        'Расстояние до школы (м)', 
        'Расстояние до детсада (м)', 
        'Расстояние до детской поликлиники (м)', 
        'Расстояние до доп. образования (м)',
        'Расстояние до взрослой поликлиники (м)',
        'Кол-во детсадов в радиусе 1 км',  
        'Кол-во парков в радиусе 1 км', 
        'Кол-во школ в радиусе 1 км',
        'Кол-во школ доп. образования в радиусе 1 км',
        'Кол-во мед. учреждений в радиусе 1 км'
    ]

    categorical_features = [
        'Район', 'Микрорайон', 'Ремонт', 
        'Комнаты/Планировка', 'Отопление', 'Продаётся с мебелью', 
        'Вид из окон', 'Балкон/лоджия', 'Санузел', 'Тип этажа'
    ]

    
    
    all_features = categorical_features + numerical_features
    X = df[all_features].copy()
    
    # Логарифмирование таргета
    y = np.log1p(df[target_column]).copy()
    
    # Заполнение пропусков
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('Не указано')
        
    for col in numerical_features:
        if X[col].isnull().any():
            median_value = X[col].median()
            X[col] = X[col].fillna(median_value)

    print(f"🛠 Всего признаков для модели: {X.shape[1]} ({len(categorical_features)} катег., {len(numerical_features)} числовых)")
    # ============================================================================

    # [Внедрение пункта 9 и 11]: Кросс-валидация K-Fold и регуляризация против переобучения
    print("🏋️‍♂️ Запуск кросс-валидации и обучения CatBoostRegressor...")
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Списки для сбора метрик по всем фолдам
    maes, mapes, r2s = [], [], []
    
    best_r2 = -1
    best_model = None
    
    # Сброс индексов, чтобы KFold работал корректно
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Обучение на фолде {fold + 1}/{n_splits} ---")
        
        X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train_f, y_train_f, cat_features=categorical_features)
        val_pool = Pool(X_val_f, y_val_f, cat_features=categorical_features)
        
        # [Параметры пункта 11]: зажали глубину до 5, подняли l2_leaf_reg до 10
        model = CatBoostRegressor(
            iterations=1500,          
            learning_rate=0.05,       
            depth=5,                  
            l2_leaf_reg=10.0,
            loss_function='RMSE',     
            random_seed=42 + fold,
            verbose=200               
        )
        
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        
        # Предсказания для текущего фолда
        log_preds = model.predict(X_val_f)
        preds_orig = np.expm1(log_preds)
        y_val_orig = np.expm1(y_val_f)
        
        # Метрики фолда
        f_mae = mean_absolute_error(y_val_orig, preds_orig)
        f_mape = mean_absolute_percentage_error(y_val_orig, preds_orig)
        f_r2 = r2_score(y_val_orig, preds_orig)
        
        maes.append(f_mae)
        mapes.append(f_mape)
        r2s.append(f_r2)
        
        print(f"Фолд {fold + 1} завершен. R²: {f_r2:.4f} | MAPE: {f_mape * 100:.2f}%")
        
        # Сохраняем лучшую модель по метрике R2
        if f_r2 > best_r2:
            best_r2 = f_r2
            best_model = model

    print("\n" + "="*40)
    print("📈 СРЕДНИЕ МЕТРИКИ КРОСС-ВАЛИДАЦИИ (5 ФОЛДОВ):")
    print(f"   Средняя абсолютная ошибка (MAE): {np.mean(maes):,.2f} ₽".replace(",", " "))
    print(f"   Средняя относительная ошибка (MAPE): {np.mean(mapes) * 100:.2f}%")
    print(f"   Коэффициент детерминации (R²): {np.mean(r2s):.4f}")
    print("="*40)

    # Анализ важности признаков у лучшей модели
    importance = best_model.get_feature_importance()
    feature_importances = pd.DataFrame({
        'Признак': all_features, 
        'Важность (%)': importance
    }).sort_values(by='Важность (%)', ascending=False)
    
    print("\n📊 ТОП ВАЖНЫХ ПРИЗНАКОВ ДЛЯ ЦЕНООБРАЗОВАНИЯ (ЛУЧШИЙ ФОЛД):")
    print(feature_importances.head(len(all_features)).to_string(index=False))
    print("="*40)

    # Сохранение лучшей версии модели
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/house_price_model.pkl"
    joblib.dump(best_model, model_save_path)
    print(f"💾 Лучшая модель успешно сохранена по пути: {model_save_path}\n")

if __name__ == "__main__":
    train_and_save_model()