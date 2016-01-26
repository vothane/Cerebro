(ns cerebro.RestrictedBoltzmannMachine.RBM
  (:use [cerebro.Utils.utils]))

; Boltzmann Machines [BMs] are a particular form of energy-based model which
; contain hidden variables. Restricted Boltzmann Machines further restrict BMs
; to those without visible-visible and hidden-hidden connections.

(defn RBM-propup [v weights bias]
  (let [_ (println weights)
         pre-sigmoid-activation (reduce + (map * weights v))
        pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-propdown [rbm h idx bias]
  (let [pre-sigmoid-activation (reduce + (map #(* (nth %1 idx) %2) (:weights rbm) h))
        pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-sample-h-given-v [rbm v0-sample]
  (let [m (map #(RBM-propup v0-sample %1 %2) (:weights rbm) (:hbias rbm))
        s (map #(binomial 1 %) m)]
    (hash-map :means m :samples s)))

(defn RBM-sample-v-given-h [rbm h0-sample]
  (let [m (map-indexed #(RBM-propdown rbm h0-sample %1 %2) (:vbias rbm))
        s (map #(binomial 1 %) m)]
    (hash-map :means m :samples s)))

(defn RBM-gibbs-hvh [rbm h0-sample]
  (let [{nv-means :means nv-samples :samples} (RBM-sample-v-given-h rbm h0-sample)
        {nh-means :means nh-samples :samples} (RBM-sample-h-given-v rbm nv-samples)]
    (hash-map :v|h (RBM-sample-v-given-h rbm h0-sample)
              :h|v (RBM-sample-h-given-v rbm nv-samples))))

(defn RBM-contrastive-divergence [rbm inputs lr k]
  (let [{ph-means :means
         ph-samples :samples} (RBM-sample-h-given-v rbm inputs)
        hvh-fn (fn [{{nh-samples :samples} :h|v}] (RBM-gibbs-hvh rbm nh-samples))
        {{nv-means :means
          nv-samples :samples} :v|h
         {nh-means :means
          nh-samples :samples} :h|v} (take k (iterate hvh-fn (RBM-gibbs-hvh rbm ph-samples)))
        weights (mapv
                  #(mapv
                     (fn [w i nvs] (+ w (/ (* lr (- (* %1 i) (* %2 nvs))) (:n rbm))))
                     %3 inputs nv-samples)
                  ph-means nh-means (:weights rbm))
        hbias (mapv #(+ (* lr (/ (- %1 %2) (:n rbm))) %3) ph-samples nh-means (:hbias rbm))
        vbias (mapv #(+ (* lr (/ (- %1 %2) (:n rbm))) %3) inputs nv-samples (:vbias rbm))]
    (hash-map :weights weights :hbias hbias :vbias vbias)))

(defn transpose [m]
  (apply mapv vector m))

(defn RBM-reconstruct [rbm v]
  (let [h (map #(RBM-propup v %1 %2) (:weights rbm) (:hbias rbm))
        activations (map #(reduce + (map % h)) (transpose (:weights rbm)))
        pre-sigmoid-activations(map + activations (:vbias rbm))]
    (map sigmoid pre-sigmoid-activations)))
  