[ns cerebro.RestrictedBoltzmannMachine.RBM
  [:use [cerebro.Utils.utils]]]

; Boltzmann Machines [BMs] are a particular form of energy-based model which
; contain hidden variables. Restricted Boltzmann Machines further restrict BMs
; to those without visible-visible and hidden-hidden connections.

(defn RBM-propup [v weights bias]
  (let [pre-sigmoid-activation (reduce + (map * weights v))
        pre-sigmoid-activation (+ pre-sigmoid-activation b)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-propdown [rbm h idx bias]
  (let [pre-sigmoid-activation (reduce + (map #(* (nth %1 idx) %2) (:weights rbm) h))
        pre-sigmoid-activation (+ pre-sigmoid-activation b)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-sample-h-given-v [rbm v0-sample]
  (let [m (map #(RBM-propup rbm v0-sample %1 %2) (:weights rbm) (:hbias rbm))
        s (map #(binomial 1 %) m)]
    (hash-map :means m :samples s)))

(defn RBM-sample-v-given-h [rbm h0-sample]
  (let [m (map-indexed #(RBM-propdown rbm h0-sample %1 %2) (:vbias rbm))
        s (map #(binomial 1 %) m)]
    (hash-map :means m :samples s)))

(defn RBM-gibbs-hvh [rbm h0-sample]
  (let [{nv-means :means nv-samples :samples} (RBM-sample-v-given-h rbm h0-sample)
        {nh-means :means nh-samples :samples} (RBM-sample-h-given-v rbm nv-samples)]
    (hash-map :v|h (RBM-sample-v-given-h rbm h0-sample nv-means)
              :h|v (RBM-sample-h-given-v rbm nv-samples nh-means))))

(defn RBM-contrastive-divergence [rbm input lr k]
	(let [{ph-mean :means ph-sample :samples} (RBM-sample-h-given-v rbm input)
        {{nv-means :means nv-samples :samples} :v|h 
         {nh-means :means nh-samples :samples} :h|v} (RBM-gibbs-hvh rbm ph-sample)
  ; IN PROGRESS      
        ]
    ))
        
(defn RBM-reconstruct [rbm v reconstructed-v]
  ;IN PROGRESS
  )