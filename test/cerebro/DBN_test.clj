(ns cerebro.DBN_test
    (:require [clojure.test :refer :all]
              [cerebro.DeepBeliefNet :refer :all]))

(def dbn
  {
   :sigmoid-layers [{:weights [[ 0.14839871643137212 -0.08501163823540675 0.051985421731801734 -0.14855205346676653 -0.044137597789180485 -0.0701731889478024] 
                               [-0.10252046344168927  0.05177738360494413 0.132389904383267    -0.11088185248031387 -0.07047144939818163   0.1342016154235016] 
                               [ 0.11659342725429114 -0.07565106509550569 0.036360063996785374 -0.08211466451905437  0.09155141306199341  -0.16083974594778425]] 
                     :bias [0 0 0]} 
                    {:weights [[ 0.1913826375359961  0.19959579863963633  -0.09572763885651328] 
                               [-0.04920529688188052 0.006828221921254152 -0.17304539780927708] 
                               [-0.1938654179145095  0.1287133626717507   -0.06540140494644048]] 
                     :bias [0 0 0]}]  
   :rbm-layers [{:N 6
                 :weights [[ 0.14839871643137212 -0.08501163823540675 0.051985421731801734 -0.14855205346676653 -0.044137597789180485 -0.0701731889478024] 
                     [-0.10252046344168927  0.05177738360494413 0.132389904383267    -0.11088185248031387 -0.07047144939818163   0.1342016154235016] 
                     [ 0.11659342725429114 -0.07565106509550569 0.036360063996785374 -0.08211466451905437  0.09155141306199341  -0.16083974594778425]] 
                 :hbias [0 0 0] 
                 :vbias [0 0 0 0 0 0]} 
                {:N 6 
                 :weights [[ 0.1913826375359961  0.19959579863963633  -0.09572763885651328] 
                     [-0.04920529688188052 0.006828221921254152 -0.17304539780927708] 
                     [-0.1938654179145095  0.1287133626717507   -0.06540140494644048]] 
                 :hbias [0 0 0] 
                 :vbias [0 0 0]}] 
   :log-layer {:W [[0 0 0] [0 0 0]] :bias [0 0] :N 6}
  })

(def training-features
  [[1 1 1 0 0 0]
   [1 0 1 0 0 0]
   [1 1 1 0 0 0]
   [0 0 1 1 1 0]
   [0 0 1 1 0 0]
   [0 0 1 1 1 0]])

(def training-predictors
  [[1 0]
   [1 0]
   [1 0]
   [0 1]
   [0 1]
   [0 1]])

(def unknown-features-X
  [[1 1 0 0 0 0]
   [1 1 1 1 0 0]
   [0 0 0 1 1 0]
   [0 0 1 1 1 0]])
                    
(def outcomes-Y
  [[1 0]
   [1 0]
   [1 0]
   [0 1]
   [0 1]
   [0 1]])

(deftest DBN-test
  (testing "DBN should make accurate predictions after training"
    (let [dbn (DBN-pretrain dbn training-features 1000 0.1 1)
          dbn (finetune dbn training-features training-predictors 500 0.1)
          ];_ (println (predict dbn [1 1 0 0 0 0]))]
      (is (= (predict dbn [1 1 0 0 0 0]) "This will always fail.")))))

