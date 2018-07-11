# Shape Saliency

## Local
* Itti orientation
    * Itti - len vetva orientacie
* Itti intensity
    * Itti - len vetva intenzity
* Centroid intra
    * Vypocet pre kazdy objekt extra
    * Deskripcia tvaru - vzdialenosti od taziska (normalizacia podla maxima)
    * Gaussova pyramida deskripcie tvaru
    * Rozdiel Gaussianov - center a surround vrstvy
        * Center = 0,1,2,3,4
        * Surround = center + delta, delta = 3,4
    * Cim vacsie rozdiely vrstiev pyramidy, tym vacsia saliency bodu kontury
    * Saliency bodu kontury sa uplatni v oblasti (trojuholniku) ohranicenej bodom kontury, nasledovnym bodom kontury a centroidom tvaru - vykreslovanie saliency objektu po trojuholnikoch
    * Gaussov filter (zjemnenie prechodov medzi trojuholnikmi)
    * Normalizacia min-max
* Spectral centroid intra
    * Vypocet pre kazdy objekt extra
    * Deskripcia tvaru - vzdialenosti od taziska (normalizacia podla maxima)
    * DFT signatury
    * Magnituda spektra - fourierove deskriptory (FD)
    * Pouzitie len N/2 FDs (FD1 … FDN/2) - symetria pri 1d DFT
    * Normalizacia komponentov podla FD0
    * Porovnat FD s FD ineho tvaru cez euklidovsku vzdialenost
    * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.73.5993

## Global
* Aspect ratio
    * Pomer medzi vacsou a mensou stranou ohranicujuceho obdlznika (normalizacia podla maxima)
    * Pre vsetky pocty objektov
* Aspect ratio distance
    * Priemerny rozdiel pomerov medzi vacsou a mensou stranou ohranicujuceho obdlznika (normalizacia min-max)
    * Pre >2 objektov
* Eccentricity
    * Pomer medzi hlavnou a vedlajsou osou minimalne ohranicujucej elipsy (normalizacia podla maxima)
    * Pre vsetky pocty objektov
* Eccentricity distance
    * Priemerny rozdiel pomerov medzi hlavnou a vedlajsou osou minimalne ohranicujucej elipsy (normalizacia min-max)
    * Pre >2 objektov
* Extent
    * Pomer medzi velkostou objektu a ohranicujucim obdlznikom (normalizacia podla maxima)
    * Saliency = 1 - extent (napadnejsi je menej pravidelny objekt)
    * Pre vsetky pocty objektov
* Extent distance
    * Priemerny rozdiel pomerov medzi velkostou objektu a ohranicujucim obdlznikom (normalizacia min-max)
    * Pre >2 objektov
* Rectangularity
    * Pomer medzi velkostou objektu a minimalne ohranicujucim obdlznikom (normalizacia podla maxima)
    * Saliency = 1 - rectangularity (napadnejsi je menej pravidelny objekt)
    * Pre vsetky pocty objektov
* Rectangularity distance
    * Priemerny rozdiel pomerov medzi velkostou objektu a minimalne ohranicujucim obdlznikom (normalizacia min-max)
    * Pre >2 objektov
* Solidity
    * Pomer medzi velkostou tvaru a konvexnym obalom (convex hull) (normalizacia podla maxima)
    * Saliency = 1 - solidity (napadnejsi je menej pravidelny objekt)
    * Pre vsetky pocty objektov
* Solidity distance
    * Priemerny rozdiel pomerov medzi velkostou tvaru a konvexnym obalom (convex hull) (normalizacia min-max)
    * Pre >2 objektov
* Equivalent diameter
    * Priemer kruhu s velkostou rovnou velkosti objektu (normalizacia podla maxima)
    * Pre vsetky pocty objektov
* Equivalent diameter distance
    * Priemerny rozdiel priemerov kruhu s velkostou rovnou velkosti objektu (normalizacia min-max)
    * Pre >2 objektov
* Area
    * Velkost/obsah objektu (normalizacia podla maxima)
    * Pre vsetky pocty objektov
* Area distance
    * Priemerny rozdiel velkosti objektu (normalizacia min-max)
    * Pre >2 objektov
* Perimeter
    * Obvod objektu (normalizacia podla maxima)
    * Pre vsetky pocty objektov
* Perimeter distance
    * Priemerny rozdiel obvodov objektu (normalizacia min-max)
    * Pre >2 objektov
* Circularity ratio
    * 4*PI*velkost objektu / obvod objektu^2 (normalizacia podla maxima)
    * Saliency = 1 - circularity ratio (napadnejsi je menej pravidelny objekt)
    * Pre vsetky pocty objektov
* Circularity ratio distance
    * Priemerny rozdiel circularity ratios (normalizacia min-max)
    * Pre >2 objektov
* Hu moments
    * porovnanie vsetkych 7 Hu momentov objektu s ostatnymi - priemer (normalizacia min-max)
    * Pre >2 objektov
    * Hu moments 1:  
    * Hu moments 2:  
    * Hu moments 3:  
* Centroid inter
    * Deskripcia tvaru - vzdialenosti od taziska: invariantna voci translacii, skalovaniu a rotaciu
        * Normovanie velkosti deskripcie - 360 hodnot a normovanie hodnot - max = 1 (skalova invariancia)
        * Zoradenie hodnot od najmensej vzdialenosti od taziska (rotacna invariancia)
        * Porovnat tvar s ostatnymi ako porovnanie histogramov (deskripcii) - priemer
* Centroid inter correlation: 1 - korelacia
* Centroid inter chi square: Chi-square
* Centroid inter intersection: 1 - prienik
* Centroid inter bhattacharyya: Bhattacharyya distance
    * Cim rozdielnejsie deskripcie, tym vacsia saliency tvaru
    * Normalizacia min-max
    * Pre >2 objektov
* Spectral centroid inter
    * Deskripcia tvaru - vzdialenosti od taziska (normalizacia podla maxima)
    * DFT signatury
    * Logaritmus magnitudy spektra
    * Vyhladeny logaritmus magnitudy spektra (priemerovaci filter)
    * Rozdiel tychto 2 magnitud (spektralne reziduum)
    * IDFT s upravenou magnitudou spektra
    * Gaussov filter = saliency bodov kontury
    * Saliency bodu kontury sa uplatni v oblasti (trojuholniku) ohranicenej bodom kontury, nasledovnym bodom kontury a centroidom tvaru - vykreslovanie saliency objektu po trojuholnikoch
    * Gaussov filter (zjemnenie prechodov medzi trojuholnikmi)
    * Pre >2 objektov
* Shape context
    * Priemer vzdialenosti medzi shape contextmi
    * 200 bodov na objekt
    * Polarna sur. Sustava: 12 uhlov, 4 magnitudy
    * Pre >2 objektov
* Hausdorff distance
    * Priemer Hausdorff vzdialenosti
    * Pre >2 objektov
* Jaccard index
    * http://www.chenpingyu.org/media/VSS16_Yupei.pdf
    * Zarovnanie dvojic objektov podla centroidu
    * Jaccard index = prienik / zjednotenie objektov
    * Saliency = 1 - jaccard index
    * Pre >2 objektov
