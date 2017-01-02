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
  {:contrastive-divergence (fn [inputs lr k]
                             (let [{ph-mean :means ph-sample :samples} (sample-h-given-v hbias weights inputs)
                                   sample-means (reduce 
                                                   (fn [{{nh-sample :samples} :h|v} _] 
                                                   (gibbs-hvh hbias vbias weights nh-sample)
                                                   gibbs-hvh hbias vbias weights ph-sample) (range k))
                                   {{nv-mean :means nv-sample :samples} :v|h} sample-means
                                   {{nh-mean :means nh-sample :samples} :h|v} sample-means
                                   calc-weight (fn [weights i j]
                                                  (-> (* (nth ph-mean i) (nth inputs j))
                                                      (- (* (nth nh-mean i) (nth nv-sample j)))
                                                      (* lr)
                                                      (/ n)
                                                      (+ (el weights i j))))
                                   weights (calc-weights weights calc-weight)
                                   hbias (calc-hbias ph-sample nh-mean hbias lr n)
                                   vbias (calc-vbias inputs nv-sample vbias lr n)]
                               (RBM weights hbias vbias n)))

    :-> {:weights weights :hbias hbias :vbias vbias :n n}})

    ;; RBM helper functions
    (defn propup [v weights bias]
      (let [pre-sigmoid-activation (-> (dot-product v weights)
                                       (+ bias))]
        (sigmoid pre-sigmoid-activation)))
    
    (defn propdown [h W bias idx]
      (let [weights (nth (matrix-transpose W) idx) 
            pre-sigmoid-activation (-> (dot-product h weights)
                                       (+ bias))]
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

    (defn calc-weights [weights f] 
      (reduce 
        (fn [weights [i j]] (put weights i j f)) 
        weights
        (for [i (range-rows weights) j (range-cols weights)] [i j]))) 

    (defn calc-hbias [ph-sample nh-mean hbias lr n] 
      (mapv #(+ (/ (* lr (- %1 %2)) n) %3) ph-sample nh-mean hbias))

    (defn calc-vbias [inputs nv-sample vbias lr n] 
      (mapv #(+ (/ (* lr (- %1 %2)) n) %3) inputs nv-sample vbias))