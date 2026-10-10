//
// Created by xav on 12/16/25.
//
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// Configuration carrés d'imagettes
const int RECT_WIDTH=265;
const int RECT_HEIGHT=263;
const int RECT_AREA= 69695;
const int RECT_X= 1643;
const int RECT_Y= 2802;
const String RECT_SIZE = "[265 x 263]";

Point Croix1_ref{255, 3238}; // Exemple: Bas à gauche
Point Croix2_ref{2196, 468}; // Exemple: Haut à droite

// Configuration d'Affichage
const int WIN_WIDTHdroite = 1200;
const int WIN_HEIGHTdroite = 800;
const char* IMAGE_WINDOWdroite = "Source Image";
//const char* RESULT_WINDOW = "Result window (Heatmap)";
const char* ALIGNED_WINDOWdroite = "Image Alignee (Resultat)";
const int MIN_DISTANCE_BETWEEN_CROIX = 1000;


const int WIN_WIDTH = 1200;
const int WIN_HEIGHT = 800;
const char* IMAGE_WINDOW = "Source Image";
const char* ALIGNED_WINDOW = "Image Alignee";

//Image à analyser (à changer)
string imName = "../Outils/donnees_test/s06_0001.png";
//

string templName = "../Template_croix.png";
string dossier_pictos = "../Outils/Pictogrammes/picto";
map<int, string> pictos = {
    {1,"Vagues"},{2,"Vagues verticales"},{3, "Bonhomme horizontal"},{4,"Triangle F"},
    {5,"Bonhomme droit"},{6,"Oeuf sur croix"},{7,"Panneau Attention"},{8,"Rond avec queue en tirbouchon"},
    {9,"Feu"},{10,"Carré plus"},{11,"Orage(flèche brisée)"},{12,"Voiture"},{13,"Sens interdit"},{14,"Losange P"}
};
string taille_s= "../Outils/Tailles/small.png";
string taille_m = "../Outils/Tailles/medium.png";
string taille_l = "../Outils/Tailles/large.png";
string templatestaille[3]={"../Outils/Tailles/large.png","../Outils/Tailles/medium.png","../Outils/Tailles/small.png"};


struct ResultatMatch {
    int id_zone;
    int index_picto;
    double score;
};


void SetupWindows() {
    namedWindow(IMAGE_WINDOW, WINDOW_NORMAL);
    resizeWindow(IMAGE_WINDOW, WIN_WIDTH, WIN_HEIGHT);
    namedWindow(ALIGNED_WINDOW, WINDOW_NORMAL);
    resizeWindow(ALIGNED_WINDOW, WIN_WIDTH, WIN_HEIGHT);

    namedWindow(IMAGE_WINDOWdroite, WINDOW_NORMAL);
    resizeWindow(IMAGE_WINDOWdroite, WIN_WIDTHdroite, WIN_HEIGHTdroite);
    //namedWindow(RESULT_WINDOW, WINDOW_NORMAL);
    //resizeWindow(RESULT_WINDOW, 800, 600);
    namedWindow(ALIGNED_WINDOWdroite, WINDOW_NORMAL);
    resizeWindow(ALIGNED_WINDOWdroite, WIN_WIDTHdroite, WIN_HEIGHTdroite);
}


std::vector<Mat> charger_reference_pictos(int nombre) { //nb de pictos à analyser (14 au final)
    std::vector<Mat> refs;
    for (int i = 0; i < nombre; i++) {
        string path = dossier_pictos + to_string(i + 1) + ".png";
        Mat p = imread(path);
        if (!p.empty()) {
            refs.push_back(p);
        } else {
            cerr << "Attention: Impossible de charger " << path << endl;
        }
    }
    return refs;
}
void extraire_pictogramme(Mat image_alignee) {
    //Rect zoneDesiree(240, 750, 166, 150); //pour la base d'entraînement
    Rect zoneDesiree(240, 780, 166, 150); //pour la base de test
    Mat imageExtraite = image_alignee(zoneDesiree);
    namedWindow("Image Extraite (ROI)", WINDOW_AUTOSIZE);
    imshow("Image Extraite (ROI)", imageExtraite);
}

std::vector<cv::Point> trouver_croix(cv::Mat img, cv::Mat templ) {
    Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    Mat img_display;
    img.copyTo(img_display);

    std::vector<cv::Point> centresTrouves;
    const int N_MAX_ATTEMPTS = 5;

    for (int k = 1; k <= N_MAX_ATTEMPTS; k++) {
        Point minLoc, maxLoc, matchLoc;
        double minVal, maxVal;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        matchLoc = maxLoc;
        result.at<float>(maxLoc.y, maxLoc.x) = 0.0; // on masque pour la prochaine itération

        Point centre_candidat = Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2);

        if (centresTrouves.empty()) {
            centresTrouves.push_back(centre_candidat);
        }
        else if (centresTrouves.size() == 1) {
            Point centre1 = centresTrouves[0];
            double distance = std::sqrt(std::pow(centre_candidat.x - centre1.x, 2) + std::pow(centre_candidat.y - centre1.y, 2));
            if (distance >= MIN_DISTANCE_BETWEEN_CROIX) {
                centresTrouves.push_back(centre_candidat);
                break;
            }
        }
    }

    // Affichage debug des croix trouvées
    for (const auto& pt : centresTrouves) {
        rectangle(img_display, Point(pt.x - templ.cols / 2, pt.y - templ.rows / 2),
            Point(pt.x + templ.cols / 2, pt.y + templ.rows / 2), Scalar(0, 0, 255), 2);
    }
    imshow(IMAGE_WINDOW, img_display);

    return centresTrouves;
}

bool aligner_image(const Mat& src, Mat& dst, const vector<Point>& centresTrouves) {
    if (centresTrouves.size() < 2) return false;

    // Tri vertical
    Point centre_bas, centre_haut;
    if (centresTrouves[0].y > centresTrouves[1].y) {
        centre_bas = centresTrouves[0];
        centre_haut = centresTrouves[1];
    } else {
        centre_bas = centresTrouves[1];
        centre_haut = centresTrouves[0];
    }

    std::vector<Point2f> srcPoints = { Point2f(centre_bas), Point2f(centre_haut) };
    std::vector<Point2f> dstPoints = { Point2f(Croix1_ref), Point2f(Croix2_ref) };

    Mat warp_mat = estimateRigidTransform(srcPoints, dstPoints, false);
    if (warp_mat.empty()) return false;

    warpAffine(src, dst, warp_mat, src.size());
    return true;
}

// Fonction analyse et affichage console
std::vector<string> analyser_pictogrammes(Mat img_alignee, const vector<Mat>& ref_pictos) {

    std::vector<ResultatMatch> liste_resultats;
    int x_fixe = 223, y_depart = 750, largeur = 200, hauteur = 231, pas_vertical = 344;

    cout << " Résultat analyse des 7 lignes :" << endl;


    for (int i = 0; i < 7; i++) {
        ResultatMatch res_actuel = { i, -1, -1.0 };
        int y_courant = y_depart + (pas_vertical * i);

        Rect zone_a_tester(x_fixe, y_courant, largeur, hauteur);
        Rect limitesImage(0, 0, img_alignee.cols, img_alignee.rows);
        Rect zoneSecurisee = zone_a_tester & limitesImage;

        if (zoneSecurisee.area() > 0) {
            Mat image_zone = img_alignee(zoneSecurisee);

            for (int j = 0; j < ref_pictos.size(); j++) {
                Mat picto_ref = ref_pictos[j];
                if (picto_ref.rows > image_zone.rows || picto_ref.cols > image_zone.cols)
                    resize(picto_ref, picto_ref, Size(image_zone.cols, image_zone.rows));

                Mat res_match;
                matchTemplate(image_zone, picto_ref, res_match, TM_CCOEFF_NORMED);
                double minVal, maxVal;
                minMaxLoc(res_match, &minVal, &maxVal);

                if (maxVal > res_actuel.score) {
                    res_actuel.score = maxVal;
                    res_actuel.index_picto = j;
                }
            }
        }
        liste_resultats.push_back(res_actuel);


        cout << "Ligne " << i + 1 << " : ";
        if (res_actuel.index_picto != -1) {
            cout << "Picto " << res_actuel.index_picto + 1 << " ie " << pictos[res_actuel.index_picto+1];
                 //<< " | Confiance: " << int(res_actuel.score * 100) << "%";   //visiblement pas besoin d'afficher cela juste trouvé/non

            if (res_actuel.score < 0.6) cout << " [Reconnaissance douteuse]";
        } else {
            cout << "Non reconnu / Hors zone";
        }
        cout << endl;
    }

    vector<int> indices_seuls;
    vector <string> names_seuls;

    for (const auto& res : liste_resultats) {
        for (int i=0; i<5; i++)
            indices_seuls.push_back(res.index_picto+1);
    }
    for (const auto& res : liste_resultats) {
        for (int i=0; i<5; i++)
            names_seuls.push_back(pictos[res.index_picto+1]);
    }

    return names_seuls;
}

vector<string> analyser_formulaire_complet(string chemin_image, string chemin_template, string dossier_pictos) {


    SetupWindows();


    Mat img = imread(chemin_image);
    Mat templ = imread(chemin_template);
    vector<Mat> ref_pictos = charger_reference_pictos(14);

    if (img.empty() || templ.empty() || ref_pictos.empty()) {
        cerr << "Erreur critique : Une ou plusieurs images sont introuvables." << endl;
        return {}; // Retourne un vecteur vide en cas d'erreur
    }

    // Détection des croix
    vector<Point> centres = trouver_croix(img, templ);

    // Alignement
    Mat img_alignee;
    bool alignement_ok = aligner_image(img, img_alignee, centres);

    if (!alignement_ok) {
        cerr << "Echec de l'alignement (Croix non trouvées ou calcul impossible)." << endl;
        return {};
    }

    // Affichage du résultat aligné
    imshow(ALIGNED_WINDOW, img_alignee);

    // Analyse et retour des résultats
    return analyser_pictogrammes(img_alignee, ref_pictos);
}


vector<Point> MatchingMethoddroite(Mat img,Mat templ, int match_method)
{
    resize(templ, templ, Size(150,150));
    Mat result;
    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create( result_rows, result_cols, CV_32FC1 );

    matchTemplate( img, templ, result, match_method );
    //normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

    Mat img_display;
    img.copyTo( img_display );

    vector<Point> centresTrouves;
    vector<Point> coinsTrouves;

    // 5 meilleurs points puis filtrage
    const int N_MAX_ATTEMPTS = 5;

    for(int k = 1; k <= N_MAX_ATTEMPTS; k++)
    {
        Point minLoc; Point maxLoc;
        Point matchLoc;
        double minVal; double maxVal;

        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        const double THRESHOLD_CORRESPONDANCE=0.25;
        double score;
        if( match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED )
        {
            matchLoc = minLoc;
            score = minVal;
        }
        else
        {
            matchLoc = maxLoc;
            score = maxVal;
        }
        bool matchValide = (match_method == TM_SQDIFF|| match_method ==TM_SQDIFF_NORMED)
            ? (score <=(THRESHOLD_CORRESPONDANCE))
            : (score >= THRESHOLD_CORRESPONDANCE);

        if (!matchValide) {
            break;
        }
        rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0, 0, 255), 2, 8, 0 );
        result.at<float>(matchLoc.y, matchLoc.x)=(match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)? 1.0f : 0.0f;
        Point centre_candidat = Point(matchLoc.x + templ.cols / 2, matchLoc.y + templ.rows / 2);


        if (centresTrouves.empty()) {
            centresTrouves.push_back(centre_candidat);
        } else if (centresTrouves.size() == 1) {

            Point centre1 = centresTrouves[0];
            double distance = std::sqrt(std::pow(centre_candidat.x - centre1.x, 2) + std::pow(centre_candidat.y - centre1.y, 2));

            if (distance >= MIN_DISTANCE_BETWEEN_CROIX) {
                centresTrouves.push_back(centre_candidat);
                break;
            }
        }
    }

    Mat result_display;
    result.convertTo(result_display, CV_8U, 255.0);

    //imshow( IMAGE_WINDOWdroite, img_display );
    //imshow( RESULT_WINDOW, result_display );

    return centresTrouves;
}

// comparaison de 2 contours détectés
bool comparerContours(const std::vector<Point>& a, const std::vector<Point>& b) {
    Rect ra = boundingRect(a);
    Rect rb = boundingRect(b);

    if (std::abs(ra.y - rb.y) < 50) {
        return ra.x < rb.x;
    }
    // Sinon, on trie par Y (haut en bas)
    return ra.y < rb.y;
}


Mat mainimagedroite(string imNamedroite)
{

    Mat img = imread(imNamedroite);
    Mat templ = imread(templName);

    if (img.empty() || templ.empty()) {
        cerr << "Erreur: Impossible de charger l'image ou le modèle." << endl;
        return cv::Mat();
    }


    //int match_method = TM_CCOEFF_NORMED;

    vector<Point> centresTrouves = MatchingMethoddroite( img, templ, TM_SQDIFF_NORMED );

    if (centresTrouves.size() < 2) {
        cerr << "Erreur: Moins de deux croix detectees. Alignement impossible ." << endl;
        waitKey(0);
        return cv::Mat();
    }

    Point centre_bas;
    Point centre_haut;

    if (centresTrouves[0].y > centresTrouves[1].y)
    {
        centre_bas = centresTrouves[0];
        centre_haut = centresTrouves[1];
    }
    else
    {
        centre_bas = centresTrouves[1];
        centre_haut = centresTrouves[0];
    }

    Point2f centre1_triee = centre_bas;
    Point2f centre2_triee = centre_haut;

    // cout << "Centre 1 trie (BAS Y) : " << centre1_triee.x << "," << centre1_triee.y << endl;
    // cout << "Centre 2 trie (HAUT Y) : " << centre2_triee.x << "," << centre2_triee.y << endl;
    //cout << "Point croix de référence 1 : " << Croix1_ref.x << "," << Croix1_ref.y << endl;
    //cout << "Point croix de référence 2 : " << Croix2_ref.x << "," << Croix2_ref.y << endl;

    std::vector<Point2f> srcPoints;
    std::vector<Point2f> dstPoints;

    srcPoints.push_back(centre1_triee);
    srcPoints.push_back(centre2_triee);

    dstPoints.push_back(Croix1_ref);
    dstPoints.push_back(Croix2_ref);

    //Mat warp_mat = estimateRigidTransform(srcPoints, dstPoints, false);
	Mat warp_mat = estimateAffinePartial2D(srcPoints, dstPoints);

    if (warp_mat.empty()) {
        cerr << "Erreur: Impossible de calculer la matrice de transformation." << endl;
        waitKey(0);
        return cv::Mat();
    }

    Mat image_alignee;
    warpAffine(img, image_alignee, warp_mat, img.size());

    //imshow(ALIGNED_WINDOWdroite, image_alignee);

    //waitKey(0);
    return image_alignee;
}

void affiche_contours(Mat image, vector<vector<Point>> contours) {
	for (int i=150;i<contours.size();i++){
		drawContours(image,contours,i,Scalar(0,255,0),3);
	}
}

vector<Mat> charger_templates_tailles_init() {
    vector<Mat> templates;
    for (int i = 0; i < 3; i++) {
        // On charge en niveaux de gris comme dans ton code original
        Mat t = imread(templatestaille[i], IMREAD_GRAYSCALE);
        if (t.empty()) {
            cerr << "Erreur critique: Impossible de charger le template taille " << templatestaille[i] << endl;
            exit(-1); // On arrête tout si un fichier manque
        }
        templates.push_back(t);
    }
    return templates;
}

int emplacementstaille[7][4]={{36*5, 178*5, 59*5, 27*5},{ 34*5, 248*5, 62*5, 25*5},{ 37*5, 316*5, 57*5, 25*5},{ 32*5, 388*5, 70*5, 24*5},{ 37*5, 458*5, 57*5, 22*5},{ 40*5, 529*5, 53 *5,19*5},{ 43*5, 596*5, 51*5, 23*5}};
string rendretailleaux(Mat img, int x, int y, int width, int height, int matchingmethod, const vector<Mat>& templates_refs) {
    Rect roi(x, y, width, height);

    // Vérification des limites pour éviter un crash si le ROI sort de l'image
    if (x + width > img.cols || y + height > img.rows) return "erreur_dim";

    Mat cropped = img(roi).clone();
    Scalar meanValuescal = mean(cropped);
    double meanValue = meanValuescal[0];


    if (meanValue > 252) {
        return "empty";
    }
    else {
        Mat result;
        double maxscore;
        if (matchingmethod == TM_SQDIFF || matchingmethod == TM_SQDIFF_NORMED) {
            maxscore = DBL_MAX;
        } else {
            maxscore = -DBL_MAX;
        }

        double score, minVal, maxVal;
        Point minLoc, maxLoc;
        int indice = -1;

        // Boucle sur les 3 templates PRÉ-CHARGÉS
        for (int i = 0; i < 3; i++) {
            Mat templ = templates_refs[i];

            if (templ.cols > cropped.cols || templ.rows > cropped.rows) continue;

            int result_cols = cropped.cols - templ.cols + 1;
            int result_rows = cropped.rows - templ.rows + 1;
            result.create(result_rows, result_cols, CV_32FC1);

            matchTemplate(cropped, templ, result, matchingmethod);
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            if (matchingmethod == TM_SQDIFF || matchingmethod == TM_SQDIFF_NORMED) {
                score = minVal;
                if (score < maxscore) {
                    indice = i;
                    maxscore = score;
                }
            } else {
                score = maxVal;
                if (score > maxscore) {
                    indice = i;
                    maxscore = score;
                }
            }
        }

        if (indice == 0) return "large";
        else if (indice == 1) return "medium";
        else if (indice == -1) return "erreur";
        else return "small"; // indice 2
    }
}

vector<string> rendretaille(Mat img, int matchingmethod, const vector<Mat>& templates_refs){
    if(img.channels() == 3) cvtColor(img, img, COLOR_BGR2GRAY);

    vector<string> res(7);
    for (int i=0; i<7; i++) {
        // On transmet 'templates_refs'
        res[i] = rendretailleaux(img, emplacementstaille[i][0], emplacementstaille[i][1],
                                 emplacementstaille[i][2], emplacementstaille[i][3],
                                 matchingmethod, templates_refs);
    }
    return res;
}

int tailles[12][4]={{2202,613,35,31},{2135,611,41,33},{2091,610,30,31},{2034,610,40,34},{1979,607,39,39},{1924,608,40,38},{1869,607,37,41},{1812,610,39,33},{1756,607,38,39},{1697,611,43,34},{1644,608,38,39},{1588, 612, 38, 34}};
int carrepresent(Mat img, int x, int y, int width, int height) {


    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display
    Rect roi(x,y,width,height); // x, y, width, height

    Mat cropped = img(roi).clone();
    //à enlever potentiellement
    //imshow( "Display window", cropped);
    Scalar meanValuescal = mean(cropped);
    double meanValue = meanValuescal[0];
    //à enlever potentiellement
    cout<< "valeur moyenne :"<<meanValue<<endl;
    if (meanValue>150) {
        return 1;
    }
    else {
        return 0;
    }
}
//pour les carrés : sans resize :
// 1:2202 613 35 31 ,2: 2135 611 41 33,3 : 2091 610 30 31, 4:2034 610 40 34, 5:1979 607 39 39,6:1924 608 40 38,7:1869 607 37 41,8:1812 610 39 33,9:1756 607 38 39,10:1697 611 43 34,11:1644 608 38 39
//12: 1588 612 38 34 peut etre que 12 existe pas

string numeropage(Mat img) { // faire attention à donner une image remise droite (au même endroit que l'image de référence pour les carrés de numéros
    cvtColor(img, img, COLOR_BGR2GRAY);
    int num =0;
    for (int i=0;i<12;i++) {
        num+=carrepresent(img,tailles[i][0],tailles[i][1],tailles[i][2],tailles[i][3])<<i;
    }
    int it = 5-std::to_string(num).length();
    string res = "";
    for (int i=0;i<it;i++) {
        res+="0";
    }
    res+=std::to_string(num);
    cout<< "numero rendu de l'image :"<<res<<endl;
    return res;

}

map<string,string> analyse(Mat matrice, Mat resultat, const vector<Mat>& templates_refs, int k) {
    map<string,string> map;

    //val défaut
    map["label"] = "Inconnu";

    string numpage = numeropage(resultat);

    map["form"]=numpage;
    map["scripter"]=numpage.substr(0, 3);
    map["page"]=numpage.substr(3, 2);
    map["row"]=to_string(k/5);
    map["column"]=to_string(k%5);

    // Appel avec les templates pré-chargés
    vector<string> tailles = rendretaille(resultat, 5, templates_refs);


    for (int i = 0; i < tailles.size(); i++) {
        map["size"] = tailles[i];
        map["taille"] = tailles[i]; // Ajout pour la sauvegarde qui utilise map["taille"]
    }


    static vector<string> labels = analyser_formulaire_complet(imName, templName, dossier_pictos);
    int imagette = stoi(map["row"])*5+stoi(map["column"]);
    if(imagette < labels.size()) map["label"] = labels[imagette];
    else map["label"] = "HorsIndex";

    return map;
}


void sauvegarde_dans_database(Mat matrice, map<string,string> m) {
	string nom_fichier="../DataDirectory/"+
		m["label"]+"_"+
		m["scripter"]+"_"+
		m["page"]+"_"+
		m["row"]+"_"+
		m["column"];

	imwrite(nom_fichier+".png",matrice);
	ofstream descriptionFile(nom_fichier+".txt");
	if (!descriptionFile.is_open()) {
		cerr<<"Erreur lors de la création du flux "<<nom_fichier+".txt"<<endl;
		return;
	}

	descriptionFile<<"# free comment (group name, year…) (other comment lines allowed)\n"; // A compléter ensuite
	descriptionFile<<"label"+ m["label"]+"<labelName> \n";
	descriptionFile<<"form <" << m["form"] << "> \n";
	descriptionFile<<"scripter <" << m["scripter"] << "> \n";
	descriptionFile<<"page <" << m["page"] << "> \n";
	descriptionFile<<"row <" << m["row"] << "> \n";
	descriptionFile<<"column <" << m["column"] << "> \n";
	descriptionFile<<"size <"+m["taille"]+"> \n";
	descriptionFile.close();

}




float global(string image, int nb_scan, Mat resultat) {
    Mat im = imread(image);
    Mat gray, bin;

    cvtColor(im, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 3);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(bin, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> contoursValides;
    int intervalle = 7;

    for (const auto& contour : contours) {
        Rect box = boundingRect(contour);
        if (box.width >= RECT_WIDTH - intervalle && box.width <= RECT_WIDTH + intervalle &&
            box.height >= RECT_HEIGHT - intervalle && box.height <= RECT_HEIGHT + intervalle) {

            contoursValides.push_back(contour);
            }
    }


    std::sort(contoursValides.begin(), contoursValides.end(), comparerContours);


    float compteur = 0.f;

    for (int k = 0; k < contoursValides.size(); k++) {
        compteur++;
        Rect box = boundingRect(contoursValides[k]);

        int marge = 6;
        Rect box2(box.x + marge, box.y + marge, box.width - 2 * marge, box.height - 2 * marge);

        // Sécurité pour ne pas sortir de l'image
        box2 = box2 & Rect(0, 0, im.cols, im.rows);

        Mat matrice = im(box2).clone();

        vector<Mat> templates_tailles = charger_templates_tailles_init();
        map<string, string> map = analyse(matrice, resultat, templates_tailles,k);
        sauvegarde_dans_database(matrice, map);
    }

    return compteur;
}

int main (void) {

    SetupWindows();

    vector<Mat> templates_tailles = charger_templates_tailles_init();

    for (int i = 1; i <= 30; i++) {
        //string nom_image = "../BDTest/" + to_string(i) + ".png";
        //string nom_image = "../BDTest/1.png";

        Mat resultat = mainimagedroite(imName);

        if (!resultat.empty()) {
            global(imName, i, resultat);
        } else {
            cerr << "Erreur alignement pour l'image " << i << endl;
        }
    }

    Mat img = imread(imName);
    Mat templ = imread(templName);
    vector<Mat> ref_pictos = charger_reference_pictos(14);

    if (img.empty() || templ.empty() || ref_pictos.empty()) {
        cerr << "Erreur fatale : Images manquantes." << endl;
        return -1;
    }

    vector<Point> centres = trouver_croix(img, templ);

    //Alignement de l'image
    Mat img_alignee;
    if (!aligner_image(img, img_alignee, centres)) {
        cerr << "Echec de l'alignement." << endl;
        return -1;
    }

    //Computation des pictogrammes trouvés
    analyser_pictogrammes(img_alignee, ref_pictos);

    imshow(ALIGNED_WINDOW, img_alignee);
	return 0;
}