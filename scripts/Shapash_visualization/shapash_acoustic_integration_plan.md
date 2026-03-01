# shapash integration plan: acoustic archeology

## 1. objective

to transform raw acoustic probability logs into a human-readable narrative. we move from "detecting sounds" to "explaining environments" by grouping correlated labels into cohesive "acoustic scenes."

---

## 2. shapash theory & mechanics

### the smartexplainer

shapash acts as a "human-centric" wrapper around complex math. it uses **shap (shapley additive explanations)** as a backend. in our context:

* **base value:** the average probability of a sound (the "background drone").
* **contributions:** how much other sounds (like "clickety-clack") pushed the probability of the target (like "train") up or down at a specific second.

### persistence & portability (avoiding "vanishing models")

by default, the trained model and shapash state live only in RAM. to make the "acoustic brain" permanent:

* **model persistence:** use `joblib` to save the raw scikit-learn model (the "knowledge").
* **explainer state:** use `xpl.save('file.pkl')` to store the entire compiled state, including calculated shap values.
* **the smartpredictor:** for the lightest export, convert the explainer to a `SmartPredictor`. this creates a portable object that can be loaded later to generate explanations for new recordings without needing the full shapash library overhead.

---

## 3. glitches to avoid (lessons learned)

### attribute mismatches (v2.x changes)

* **the glitch:** `'SmartExplainer' object has no attribute 'features_importance'`.
* **the fix:** in shapash 2.x, global importance is stored in the `features_imp` attribute. ensure you call `xpl.compute_features_import()` first to populate it. the old `to_pandas()` method can still be used for local details, but the `features_imp` attribute is the "pro way" for global metrics.

### performance bottlenecks

* **the glitch:** 1 hour+ execution time for 28k rows on mobile/termux.
* **the fix:** **strategic sampling.** train the model on the full set, but only `compile` shapash on the "interest peaks" (moments of highest probability) and a small random background sample.

---

## 4. integration steps

### step 1: automated ambiance deduction

use a correlation matrix (threshold > 0.6) to find "acoustic synonyms."

* **logic:** if sound A and sound B always fire together, they belong to the same scene.

### step 2: feature dictionary mapping

create a `features_dict` to rename technical labels to descriptive ones.

* *example:* "outside, urban or manmade" -> "city street texture."

### step 3: compiling the grouped explainer

initialize `SmartExplainer` with the `features_groups` dictionary.

```python
# conceptual logic
groups = {"walking_gait": ["Animal", "Horse", "Clip-clop"]}
xpl = SmartExplainer(model=model, features_groups=groups)
```

### step 4: temporal narrative export

instead of a manual loop, use the built-in `to_pandas()` parameters to extract the "Top Drivers" automatically.

* **the shapash way:** `xpl.to_pandas(max_contrib=3)` returns a clean dataframe where each sample is already linked to its top 3 contributing groups/features.
* **benefit:** this avoids writing complex filtering logic and ensures the "base value" (the background drone) is accounted for correctly in the ranking.

### step 5: persistence & export

save the "acoustic brain" so it can be reused on other recordings.

```python
# export the lightweight predictor
predictor = xpl.to_smartpredictor()
predictor.save("acoustic_archeology_v1.pkl")
```

---

## 5. final vision

the end result is a script that doesn't just say "train detected at 76s." it says:

> **at 76.4s, the "tramway ambiance" scene peaked. the primary driver was the rhythmic "clickety-clack" of the wheels.**
