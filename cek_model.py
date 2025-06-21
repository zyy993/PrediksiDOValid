import pickle

with open('model/model_numpy.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model berhasil dimuat:", type(model))

with open('model/selector.pkl', 'rb') as f:
    selector = pickle.load(f)

print("✅ Selector berhasil dimuat:", type(selector))
