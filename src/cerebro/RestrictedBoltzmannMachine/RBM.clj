(ns cerebro.RestrictedBoltzmannMachine.RBM
  (:use [cerebro.Utils.utils]))

; Boltzmann Machines [BMs] are a particular form of energy-based model which
; contain hidden variables. Restricted Boltzmann Machines further restrict BMs
; to those without visible-visible and hidden-hidden connections.

;; API functions
(defn contrastive-divergence [rbm inputs lr k] ((:contrastive-divergence rbm) inputs lr k))
(defn rbm->map [rbm] (:->map rbm))

(declare sample-h-given-v)
(declare gibbs-hvh)
(declare calc-weights) 
(declare calc-hbias)
(declare calc-vbias)

(defn RBM [weights hbias vbias n]
  (let [{ph-mean :means ph-sample :samples} (sample-h-given-v hbias weights inputs)
        sample-means (reduce 
                        (fn [{{nh-sample :samples} :h|v} _] 
                        (gibbs-hvh hbias vbias weights nh-sample)
                        gibbs-hvh hbias vbias weights ph-sample) (range k))
        {{nv-mean :means nv-sample :samples} :v|h} sample-means
        {{nh-mean :means nh-sample :samples} :h|v} sample-means]

    {:contrastive-divergence (fn [inputs lr k]
                               (let [weights (calc-weighrs ph-mean nh-mean weights)
                                     hbias (calc-hbias ph-sample nh-mean hbias)
                                     vbias (calc-vbias inputs nv-sample vbias)]
                                 (RBM weights hbias vbias n)))

    :-> {:weights weights :hbias hbias :vbias vbias :n n}
   ))

    ;; RBM helper functions
    (defn propup [v weights bias]
      (let [pre-sigmoid-activation (reduce + (map #(* %1 %2) v weights))
            pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
        (sigmoid pre-sigmoid-activation)))
    
    (defn propdown [h W bias idx]
      (let [weights                (nth (matrix-transpose W) idx) 
            pre-sigmoid-activation (reduce + (map #(* %1 %2) h weights)) 
            pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
        (sigmoid pre-sigmoid-activation)))
    
    (defn sample-h-given-v [hbias W v0-sample]
      (let [m (mapv #(propup v0-sample %1 %2) W hbias)
            s (mapv #(binomial 1 %) m)]
        {:means m :samples s}))
    
    (defn sample-v-given-h [vbias W h0-sample]
      (let [m (map-indexed #(propdown h0-sample W %2 %1) vbias)
            s (mapv #(binomial 1 %) m)]
        {:means (vec m) :samples s}))
    
    (defn gibbs-hvh [hbias vbias W h0-sample]
      (let [s-v|h (sample-v-given-h vbias W h0-sample)]
        {:v|h s-v|h :h|v (sample-h-given-v hbias W (:samples s-v|h))}))
 

    (defn calc-weights [ph-mean nh-mean weights] 
      (mapv 
        (fn [ph-mean_i nh-mean_i W_i] 
          (mapv 
            (fn [W_ij nv-sample_j inputs_j] 
              (+ W_ij (/ (* lr (- (* ph-mean_i inputs_j) (* nh-mean_i nv-sample_j))) n)))
            W_i nv-sample inputs))
        ph-mean nh-mean weights))

    (defn calc-hbias [ph-sample nh-mean hbias] 
      (mapv #(+ (/ (* lr (- %1 %2)) n) %3) ph-sample nh-mean hbias))

    (defn calc-vbias [inputs nv-sample vbias] 
      (mapv #(+ (/ (* lr (- %1 %2)) n) %3) inputs nv-sample vbias))