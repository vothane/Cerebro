(ns cerebro.RestrictedBoltzmannMachine.RBM
  (:use [cerebro.Utils.utils]))

; Boltzmann Machines [BMs] are a particular form of energy-based model which
; contain hidden variables. Restricted Boltzmann Machines further restrict BMs
; to those without visible-visible and hidden-hidden connections.

(defn RBM-propup [v weights bias]
  (let [pre-sigmoid-activation (reduce + (map #(* %1 %2) v weights))
        pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-propdown [h W bias idx]
  (let [weights                (nth (matrix-transpose W) idx) 
        pre-sigmoid-activation (reduce + (map #(* %1 %2) h weights)) 
        pre-sigmoid-activation (+ pre-sigmoid-activation bias)]
    (sigmoid pre-sigmoid-activation)))

(defn RBM-sample-h-given-v [hbias W v0-sample]
  (let [m (mapv #(RBM-propup v0-sample %1 %2) W hbias)
        s (mapv #(binomial 1 %) m)]
    {:means m :samples s}))

(defn RBM-sample-v-given-h [vbias W h0-sample]
  (let [m (map-indexed #(RBM-propdown h0-sample W %2 %1) vbias)
        s (mapv #(binomial 1 %) m)]
    {:means (vec m) :samples s}))

(defn RBM-gibbs-hvh [hbias vbias W h0-sample]
  (let [s-v|h (RBM-sample-v-given-h vbias W h0-sample)]
    {:v|h s-v|h :h|v (RBM-sample-h-given-v hbias W (:samples s-v|h))}))

(defn RBM-contrastive-divergence [rbm inputs lr k]
  (let [{hbias :hbias vbias :vbias W :weights n :N} rbm
        {ph-mean :means ph-sample :samples} (RBM-sample-h-given-v hbias W inputs)
        {{nv-mean :means nv-sample :samples} :v|h
         {nh-mean :means nh-sample :samples} :h|v} (reduce 
                                                       (fn [{{nh-sample :samples} :h|v} _] 
                                                         (RBM-gibbs-hvh hbias vbias W nh-sample)) 
                                                       (RBM-gibbs-hvh hbias vbias W ph-sample) (range k))
        weights (mapv
                  (fn [ph-mean_i nh-mean_i W_i] 
                    (mapv
                      (fn [W_ij nv-sample_j inputs_j] (+ W_ij (/ (* lr (- (* ph-mean_i inputs_j) (* nh-mean_i nv-sample_j))) n)))
;this.W[i][j] += lr * (ph_mean[i] * float64(input[j]) - nh_means[i] * float64(nv_samples[j])) / float64(this.N)
                      W_i nv-sample inputs))
                  ph-mean nh-mean W)
        hbias (mapv #(+ (/ (* lr (- %1 %2)) n) %3) ph-sample nh-mean hbias)
        ;this.hbias[i] += lr * (float64(ph_sample[i]) - nh_means[i]) / float64(this.N)
        vbias (mapv #(+ (/ (* lr (- %1 %2)) n) %3) inputs nv-sample vbias)]
        ;this.vbias[i] += lr * float64(input[i] - nv_samples[i]) / float64(this.N)
    (assoc rbm :weights weights :hbias hbias :vbias vbias)))

