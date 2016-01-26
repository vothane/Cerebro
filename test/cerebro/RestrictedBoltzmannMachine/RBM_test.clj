(ns cerebro.RestrictedBoltzmannMachine.RBM_test
  (:require [clojure.test :refer :all]
            [cerebro.RestrictedBoltzmannMachine.RBM :refer :all]))
 
(deftest make-log-reg-test
  (testing "logistic regression"
    (let [data [[1 1 1 0 0 0] 
                [1 0 1 0 0 0] 
                [1 1 1 0 0 0] 
                [0 0 1 1 1 0] 
                [0 0 1 1 0 0] 
                [0 0 1 1 1 0]]
          W    [[ 0.06548973 -0.07128689]
                [-0.09104952  0.01710492]
                [ 0.07315632 -0.02563118]
                [ 0.16025473  0.06160991]
                [-0.00635603 -0.03596083]
                [-0.05227399  0.0763499 ]]
          vbias [0.0 0.0 0.0 0.0 0.0 0.0]
          hbias [0.0 0.0]
          rbm   {:weights W :vbias vbias :hbias hbias}
          v     [[1  1  0  0  0  0] 
                 [0  0  0  1  1  0]]
          rbm-f (fn [rbm] (RBM-contrastive-divergence rbm data 0.1 1))
          rbm   (take 1000 (iterate (rbm-f rbm)))]
      (is (= (RBM-reconstruct rbm v)
             [[ 0.99721552  0.61943291  0.99464744  0.00180803  0.00362361  0.00316299]
              [ 0.00373956  0.0027835   0.9946441   0.99817361  0.70915455  0.00214715]])))))
