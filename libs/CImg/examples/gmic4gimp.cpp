/*
 #
 #  File        : gmic4gimp.cpp
 #                ( C++ source file )
 #
 #  Description : G'MIC for GIMP - A plug-in to allow the use
 #                of G'MIC commands in GIMP.
 #                This file is a part of the CImg Library project.
 #                ( http://cimg.sourceforge.net )
 #
 #  Copyright   : David Tschumperle (GREYCstoration API)
 #
 #  License     : CeCILL v2.0
 #                ( http://www.cecill.info/licences/Licence_CeCILL_V2-en.html )
 #
 #  This software is governed by the CeCILL  license under French law and
 #  abiding by the rules of distribution of free software.  You can  use,
 #  modify and/ or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and  rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and, more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
 #
*/

// Include necessary header files.
//--------------------------------
#define cimg_display_type 0
#include "gmic.h"
#if !defined(__MACOSX__)  && !defined(__APPLE__)
#include <pthread.h>
#endif
#include <locale>
#include <gtk/gtk.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>
extern char data_gmic_def[];
extern unsigned int size_data_gmic_def;
extern unsigned char data_gmic_logo[];
extern unsigned int size_data_gmic_logo;
using namespace cimg_library;

// Define plug-in global variables.
//---------------------------------
CImgList<char> gmic_entries;             // The list of recognized G'MIC menu entries.
CImgList<char> gmic_1stlevel_entries;    // The treepath positions of 1st-level G'MIC menu entries.
CImgList<char> gmic_commands;            // The list of corresponding G'MIC commands to process the image.
CImgList<char> gmic_preview_commands;    // The list of corresponding G'MIC commands to preview the image.
CImgList<char> gmic_arguments;           // The list of corresponding needed filter arguments.
bool return_create_dialog;               // Return value of the 'create_gui_dialog()' function (set by events handlers).
void **event_infos;                      // Infos that are passed to the GUI callback functions.
char *gmic_macros = 0;                   // The array of (customized) G'MIC macros.
int image_id = 0;                        // The image concerned by the plug-in execution.
GtkTreeStore *treeview_store = 0;        // The list of the filters as a GtkTreeView model.
GimpDrawable *drawable_preview = 0;      // The drawable used by the preview window.
GtkWidget *left_pane = 0;                // The left pane of the dialog window, containing the preview window.
GtkWidget *gui_preview = 0;              // The preview window itself.
GtkWidget *treemode_stockbutton = 0;     // A temporary stock button for the expand/collapse button.
GtkWidget *treemode_button = 0;          // Expand/Collapse button for the treeview.
GtkWidget *input_combobox = 0;           // The "input layers" combobox
GtkWidget *output_combobox = 0;          // The "output mode" combobox
GtkWidget *preview_combobox = 0;         // The "output preview" combobox
GtkWidget *verbosity_combobox = 0;       // The "output messages" combobox

#define gmic_xstr(x) gmic_str(x)
#define gmic_str(x) #x
#define gmic_update_server "http://www.greyc.ensicaen.fr/~dtschump/"  // The filters update server address.
#define gmic_update_file "gmic_def." gmic_xstr(gmic_version)          // The filters update filename.

// Set/get plug-in persistent variables, using GIMP {get,set}_data() features.
//-----------------------------------------------------------------------------

// Set/get the indice of the currently selected filter.
void set_current_filter(const unsigned int current_filter) {
  const unsigned int ncurrent_filter = current_filter>gmic_entries.size?0:current_filter;
  gimp_set_data("gmic_current_filter",&ncurrent_filter,sizeof(unsigned int));
}

unsigned int get_current_filter() {
  unsigned int current_filter = 0;
  gimp_get_data("gmic_current_filter",&current_filter);
  if (current_filter>gmic_entries.size) current_filter = 0;
  return current_filter;
}

// Set/get the number of parameters of the specified filter.
void set_filter_nbparams(const unsigned int filter, const unsigned int nbparams) {
  char s_tmp[256] = { 0 };
  std::sprintf(s_tmp,"gmic_filter%u_nbparams",filter);
  gimp_set_data(s_tmp,&nbparams,sizeof(unsigned int));
}

unsigned int get_filter_nbparams(const unsigned int filter) {
  char s_tmp[256] = { 0 };
  std::sprintf(s_tmp,"gmic_filter%u_nbparams",filter);
  unsigned int nbparams = 0;
  gimp_get_data(s_tmp,&nbparams);
  return nbparams;
}

// Set/get one particular parameter of a filter, or reset all filters parameters.
void set_filter_parameter(const unsigned int filter, const unsigned int n, const char *const param) {
  char s_tmp[256] = { 0 };
  std::sprintf(s_tmp,"gmic_filter%u_parameter%u",filter,n);
  gimp_set_data(s_tmp,param,std::strlen(param)+1);
}

const char *get_filter_parameter(const unsigned int filter, const unsigned int n) {
  char s_tmp[256] = { 0 };
  std::sprintf(s_tmp,"gmic_filter%u_parameter%u",filter,n);
  static char res[4096] = { 0 };
  res[0] = 0;
  gimp_get_data(s_tmp,res);
  return res;
}

void reset_filters_parameters() {
  const char empty[] = { 0 };
  for (unsigned int i = 1; i<gmic_entries.size; ++i)
    for (unsigned int j = 0; ; ++j) {
      const char *const val = get_filter_parameter(i,j);
      if (*val) set_filter_parameter(i,j,empty); else break;
    }
}

// Set/get the filter input, output and preview and verbosity modes.
void set_input_mode(const unsigned int input_mode) {
  gimp_set_data("gmic_input_mode",&input_mode,sizeof(unsigned int));
}

unsigned int get_input_mode(bool normalized=true) {
  unsigned int input_mode = 0;
  gimp_get_data("gmic_input_mode",&input_mode);
  return normalized?(input_mode<2?1:(input_mode-2)):input_mode;
}

void set_output_mode(const unsigned int output_mode) {
  gimp_set_data("gmic_output_mode",&output_mode,sizeof(unsigned int));
}

unsigned int get_output_mode(bool normalized=true) {
  unsigned int output_mode = 0;
  gimp_get_data("gmic_output_mode",&output_mode);
  return normalized?(output_mode<2?0:(output_mode-2)):output_mode;
}

void set_preview_mode(const unsigned int preview_mode) {
  gimp_set_data("gmic_preview_mode",&preview_mode,sizeof(unsigned int));
}

unsigned int get_preview_mode(bool normalized=true) {
  unsigned int preview_mode = 0;
  gimp_get_data("gmic_preview_mode",&preview_mode);
  return normalized?(preview_mode<2?0:(preview_mode-2)):preview_mode;
}

void set_verbosity_mode(const unsigned int verbosity) {
  gimp_set_data("gmic_verbosity_mode",&verbosity,sizeof(unsigned int));
}

unsigned int get_verbosity_mode(bool normalized=true) {
  unsigned int verbosity_mode = 0;
  gimp_get_data("gmic_verbosity_mode",&verbosity_mode);
  return normalized?(verbosity_mode<2?0:(verbosity_mode-2)):verbosity_mode;
}

// Set/get the tree collapse/expand mode.
void set_treemode(const bool expand) {
  gimp_set_data("gmic_treemode",&expand,sizeof(bool));
}

bool get_treemode() {
  bool treemode = 0;
  gimp_get_data("gmic_treemode",&treemode);
  return treemode;
}

// Set/get the net update activation state.
void set_net_update(const bool net_update) {
  gimp_set_data("gmic_net_update",&net_update,sizeof(bool));
}

bool get_net_update() {
  bool net_update = true;
  gimp_get_data("gmic_net_update",&net_update);
  return net_update;
}

void set_locale() {
  char locale[8] = { 0 };
  std::sscanf(std::setlocale(LC_ALL,0),"%c%c",&(locale[0]),&(locale[1]));
  cimg::uncase(locale);
  gimp_set_data("gmic_locale",locale,std::strlen(locale)+1);
}

const char *get_locale() {
  static char res[256] = { 0 };
  res[0] = 0;
  gimp_get_data("gmic_locale",res);
  return res;
}

// Translate string into the current locale.
//------------------------------------------
#define _translate(source,dest) if (!cimg::strcmp(source,s)) { static const char *const ns = dest; return ns; }
const char *translate(const char *const s) {

  // French translation
  if (!cimg::strcmp(get_locale(),"fr")) {
    if (!s) {
      static const char *const ns = "<b>Mise &#224; jour depuis Internet impossible !</b>\n\n"
        "Merci de v&#233;rifier l'&#233;tat de votre connexion. Vous pouvez\n"
        "manuellement mettre &#224; jour vos filtres en t&#233;l&#233;chargeant :\n\n"
        "<u><small>%s%s</small></u>\n\n"
        "et en le copiant comme le fichier <i>.%s</i>\n"
        "dans votre r&#233;pertoire <i>Home</i> ou <i>Application Data</i>.";
      return ns;
    }
    _translate("_G'MIC for GIMP...","_G'MIC pour GIMP...");
    _translate("G'MIC for GIMP","G'MIC pour GIMP");
    _translate("<i>Select a filter...</i>","<i>Choisissez un filtre...</i>");
    _translate("<i>No parameters to set...</i>","<i>Pas de param&#232;tres...</i>");
    _translate("<b> Input / Output : </b>","<b> Entr&#233;es / Sorties : </b>");
    _translate("Input layers...","Calques d'entr\303\251e...");
    _translate("None","Aucun");
    _translate("Active (default)","Actif (d\303\251faut)");
    _translate("All","Tous");
    _translate("Active & below","Actif & en dessous");
    _translate("Active & above","Actif & au dessus");
    _translate("All visibles","Tous les visibles");
    _translate("All invisibles","Tous les invisibles");
    _translate("All visibles (decr.)","Tous les visibles (d\303\251cr.)");
    _translate("All invisibles (decr.)","Tous les invisibles (d\303\251cr.)");
    _translate("All (decr.)","Tous (d\303\251cr.)");
    _translate("Output mode...","Mode de sortie...");
    _translate("In place (default)","Sur place (d\303\251faut)");
    _translate("New layer(s)","Nouveau(x) calque(s)");
    _translate("New image","Nouvelle image");
    _translate("Output preview...","Mode d'aper\303\247u...");
    _translate("1st output (default)","1\303\250re image (d\303\251faut)");
    _translate("2nd output","2\303\250me image");
    _translate("3rd output","3\303\250me image");
    _translate("4th output","4\303\250me image");
    _translate("1st -> 2nd","1\303\250re -> 2\303\250me");
    _translate("1st -> 3rd","1\303\250re -> 3\303\250me");
    _translate("1st -> 4th","1\303\250re -> 4\303\250me");
    _translate("All outputs","Toutes les images");
    _translate("Output messages...","Messages de sortie...");
    _translate("Quiet (default)","Aucun message (d\303\251faut)");
    _translate("Verbose","Mode verbeux");
    _translate("Very verbose","Mode tr\303\250s verbeux");
    _translate("Debug mode","Mode d\303\251bogage");
    _translate("_Internet updates","Mises \303\240 jour _Internet");
    _translate(" Available filters (%u) :"," Filtres disponibles (%u) :");
  }

  // Catalan translation
  if (!cimg::strcmp(get_locale(),"ca")) {
    if (!s) {
      static const char *const ns =
        "<b>No ha estat possible establir una connexi&#243; a Internet !</b>\n\n"
        "Verifiqueu l'estat de la vostra connexi&#243;. Podeu\n"
        "actualitzar els vostres filtres descarregant-vos:\n\n"
        "<u><small>%s%s</small></u>\n\n"
        "i copiant-lo com a <i>.%s</i>\n"
        "a la vostra carpeta d'<i>inici</i> o la carpeta de <i>Dades de Programa</i>.";
      return ns;
    }
    _translate("_G'MIC for GIMP...","_G'MIC per al GIMP...");
    _translate("G'MIC for GIMP","G'MIC per al GIMP");
    _translate("<i>Select a filter...</i>","<i>Selecciona un filtre...</i>");
    _translate("<i>No parameters to set...</i>","<i>Sense par\303\240metres...</i>");
    _translate("<b> Input / Output : </b>","<b> Entrades / Sortides : </b>");
    _translate("Input layers...","Capes d'entrada...");
    _translate("None","Cap");
    _translate("Active (default)","Actiu (predet.)");
    _translate("All","Tots");
    _translate("Active & below","L'activa i les de sota");
    _translate("Active & above","L'activa i les de sobre");
    _translate("All visibles","Totes les visibles");
    _translate("All invisibles","Totes les invisibles");
    _translate("All visibles (decr.)","Totes les visibles (decr.)");
    _translate("All invisibles (decr.)","Totes les invisibles (decr.)");
    _translate("All (decr.)","Totes (decr.)");
    _translate("Output mode...","Mode de sortida...");
    _translate("In place (default)","A la capa actual (predet.)");
    _translate("New layer(s)","Nova/es capa/es");
    _translate("New image","Nova imatge");
    _translate("Output preview...","Previsualitzaci\303\263 de sortida...");
    _translate("1st output (default)","1era imatge (predet.)");
    _translate("2nd output","2ona imatge");
    _translate("3rd output","3era imatge");
    _translate("4th output","4rta imatge");
    _translate("1st -> 2nd","1era -> 2ona");
    _translate("1st -> 3rd","1era -> 3era");
    _translate("1st -> 4th","1era -> 4rta");
    _translate("All outputs","Totes les imatges");
    _translate("Output messages...","Missatges de sortida...");
    _translate("Quiet (default)","Sense missatges (predet.)");
    _translate("Verbose","Verb\303\263s");
    _translate("Very verbose","Molt verb\303\263s");
    _translate("Debug mode","Depuraci\303\263");
    _translate("_Internet updates","Actualitzacions per _Internet");
    _translate(" Available filters (%u) :"," Filtres disponibles (%u) :");
  }

  // Italian translation
  if (!cimg::strcmp(get_locale(),"it")) {
    if (!s) {
      static const char *const ns = "<b>Impossibile aggiornare da Internet !</b>\n\n"
        "Controllate lo stato della vostra connessione. Potete anche\n"
        "aggiornare manualmente i filtri scaricando :\n\n"
        "<u><small>%s%s</small></u>\n\n"
        "e copiandoli come il file <i>.%s</i>\n"
        "nella directory <i>Home</i> o <i>Application Data</i>.";
      return ns;
    }
    _translate("_G'MIC for GIMP...","_G'MIC per GIMP...");
    _translate("G'MIC for GIMP","G'MIC per GIMP");
    _translate("<i>Select a filter...</i>","<i>Sciegliete un Filtro...</i>");
    _translate("<i>No parameters to set...</i>","<i>Filtro senza Parametri...</i>");
    _translate("<b> Input / Output : </b>","<b> Input / Output : </b>");
    _translate("Input layers...","Input da Layers...");
    _translate("None","Nessuno");
    _translate("Active (default)","Layer Attivo (default)");
    _translate("All","Tutti");
    _translate("Active & below","Attivo & superiori");
    _translate("Active & above","Attivo & inferiori");
    _translate("All visibles","Tutti i Visibili");
    _translate("All invisibles","Tutti gli invisibili");
    _translate("All visibles (decr.)","Tutti i visibili (dal fondo)");
    _translate("All invisibles (decr.)","Tutti gli invisibili (dal fondo)");
    _translate("All (decr.)","Tutti");
    _translate("Output mode...","Tipo di output...");
    _translate("In place (default)","Applica al Layer attivo (default) ");
    _translate("New layer(s)","Nuovo(i) Layer(s)");
    _translate("New image","Nuova Immagine");
    _translate("Output preview...","Anteprima...");
    _translate("1st output (default)","Primo Output (default)");
    _translate("2nd output","Secondo Output");
    _translate("3rd output","Terzo Output");
    _translate("4th output","Quarto Output");
    _translate("1st -> 2nd","1 -> 2");
    _translate("1st -> 3rd","1 -> 3");
    _translate("1st -> 4th","1 -> 4");
    _translate("All outputs","Tutti i layers");
    _translate("Output messages...","Messaggi di Output...");
    _translate("Quiet (default)","Nessun Messaggio (default)");
    _translate("Verbose","Verbose");
    _translate("Very verbose","Messaggi Dettagliati");
    _translate("Debug mode","Debug Mode");
    _translate("_Internet updates","Aggiornamento via _Internet");
    _translate(" Available filters (%u) :"," Filtri disponibili (%u) :");
  }

  // English translation (default)
  if (!s) {
    static const char *const ns = "<b>Filters update from Internet failed !</b>\n\n"
      "Please check your Internet connection. You can\n"
      "manually update your filters by downloading :\n\n"
      "<u><small>%s%s</small></u>\n\n"
      "and copy it as the file <i>.%s</i>\n"
      "in your <i>Home</i> or <i>Application Data</i> folder.";
    return ns;
  }
  return s;
}

// Flush filter tree view
//------------------------
void flush_treeview(GtkWidget *treeview) {
  const unsigned int filter = get_current_filter();
  bool treemode = get_treemode();
  char current_path[256] = { 0 };
  unsigned int current_dir = 0;
  gimp_get_data("gmic_current_treepath",&current_path);

  if (treemode) { // Expand
    cimglist_for(gmic_1stlevel_entries,l) {
      GtkTreePath *path = gtk_tree_path_new_from_string(gmic_1stlevel_entries[l].ptr());
      gtk_tree_view_expand_row(GTK_TREE_VIEW(treeview),path,false);
      gtk_tree_path_free(path);
    }
  } else { // Collapse
    if (filter && *current_path && std::sscanf(current_path,"%u",&current_dir)==1) {
      cimglist_for(gmic_1stlevel_entries,l) {
        const char *const s_path = gmic_1stlevel_entries[l].ptr();
        unsigned int dir = 0;
        if (std::sscanf(s_path,"%u",&dir)!=1 || dir!=current_dir) {
          GtkTreePath *path = gtk_tree_path_new_from_string(gmic_1stlevel_entries[l].ptr());
          gtk_tree_view_collapse_row(GTK_TREE_VIEW(treeview),path);
          gtk_tree_path_free(path);
        }
      }
    } else gtk_tree_view_collapse_all(GTK_TREE_VIEW(treeview));
  }

  if (filter && *current_path) {
    GtkTreePath *path = gtk_tree_path_new_from_string(current_path);
    gtk_tree_view_expand_to_path(GTK_TREE_VIEW(treeview),path);
    gtk_tree_view_scroll_to_cell(GTK_TREE_VIEW(treeview),path,NULL,FALSE,0,0);
    GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));
    gtk_tree_selection_select_path(selection,path);
    gtk_tree_path_free(path);
  }

  if (treemode_stockbutton) gtk_widget_destroy(treemode_stockbutton);
  treemode_stockbutton = gtk_button_new_from_stock(treemode?GTK_STOCK_ZOOM_OUT:GTK_STOCK_ZOOM_IN);
  GtkWidget *tree_image = gtk_button_get_image(GTK_BUTTON(treemode_stockbutton));
  gtk_button_set_image(GTK_BUTTON(treemode_button),tree_image);
  gtk_widget_show(treemode_button);

  gtk_tree_view_remove_column(GTK_TREE_VIEW(treeview),gtk_tree_view_get_column(GTK_TREE_VIEW(treeview),0));
  GtkCellRenderer *renderer = gtk_cell_renderer_text_new();
  char treeview_title[1024] = { 0 };
  std::sprintf(treeview_title,translate(" Available filters (%u) :"),gmic_entries.size);
  GtkTreeViewColumn *column = gtk_tree_view_column_new_with_attributes(treeview_title,renderer,"markup",1,NULL);

  gtk_tree_view_append_column(GTK_TREE_VIEW(treeview),column);
}

// Update filter definitions : retrieve files and update treeview.
//----------------------------------------------------------------
bool update_filters_definition(const bool network_update) {

  // Free old definitions if necessary.
  if (gmic_macros) delete[] gmic_macros;
  if (treeview_store) g_object_unref(treeview_store);
  gmic_entries.assign();
  gmic_1stlevel_entries.assign();
  gmic_commands.assign();
  gmic_preview_commands.assign();
  gmic_arguments.assign();

  // Get filter definitions from the distant server.
  const char *const path_home = getenv(cimg_OS!=2?"HOME":"APPDATA");
  bool network_succeed = false;
  if (network_update) {
    char update_command[1024] = { 0 }, src_filename[1024] = { 0 }, dest_filename[1024] = { 0 };
    const char *const path_tmp = cimg::temporary_path();
    std::sprintf(src_filename,"%s/%s",path_tmp,gmic_update_file);
    std::sprintf(dest_filename,"%s/.%s",path_home,gmic_update_file);
    std::remove(src_filename);
    if (get_verbosity_mode()>0) {
#if defined(__MACOSX__)  || defined(__APPLE__)
      std::sprintf(update_command,"curl %s%s -o %s",gmic_update_server,gmic_update_file,src_filename);
#else
      std::sprintf(update_command,"wget %s%s -O %s",gmic_update_server,gmic_update_file,src_filename);
#endif
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Running update procedure, with command : '%s'\n",update_command);
    } else {
#if defined(__MACOSX__)  || defined(__APPLE__)
      std::sprintf(update_command,"curl --silent %s%s -o %s",gmic_update_server,gmic_update_file,src_filename);
#else
      std::sprintf(update_command,"wget --quiet %s%s -O %s",gmic_update_server,gmic_update_file,src_filename);
#endif
    }
    int status = cimg::system(update_command);
    status = 0;
    std::FILE *file_s = std::fopen(src_filename,"r");
    if (file_s) {
      unsigned int size_s = 0;
      std::fseek(file_s,0,SEEK_END);
      size_s = (unsigned int)std::ftell(file_s);
      std::rewind(file_s);
      if (size_s) {
        std::FILE *file_d = std::fopen(dest_filename,"w");
        char *buffer = new char[size_s], sep = 0;
        if (file_d &&
            std::fread(buffer,sizeof(char),size_s,file_s)==size_s &&
            std::sscanf(buffer,"#@gmi%c",&sep)==1 && sep=='c' &&
            std::fwrite(buffer,sizeof(char),size_s,file_d)==size_s) { network_succeed = true; std::fclose(file_d); }
        delete[] buffer;
      }
      std::fclose(file_s);
    }
  }

  // Get filter definitions from local files '.gmic' and '.gmic_def.xxxx'
  unsigned size_update = 0, size_custom = 0;
  char filename_update[1024] = { 0 }, filename_custom[1024] = { 0 };
  std::sprintf(filename_update,"%s/.gmic_def.%d",path_home,gmic_version);
  std::sprintf(filename_custom,"%s/.gmic",path_home);
  std::FILE
    *file_update = std::fopen(filename_update,"r"),
    *file_custom = std::fopen(filename_custom,"r");
  if (file_update) {
    std::fseek(file_update,0,SEEK_END);
    size_update = (unsigned int)std::ftell(file_update);
    std::rewind(file_update);
  }
  if (file_custom) {
    std::fseek(file_custom,0,SEEK_END);
    size_custom = (unsigned int)std::ftell(file_custom);
    std::rewind(file_custom);
  }
  const unsigned int size_final = size_update + size_custom + size_data_gmic_def + 1;
  char *ptrd = gmic_macros = new char[size_final];
  if (size_custom) { ptrd+=std::fread(ptrd,1,size_custom,file_custom); std::fclose(file_custom); }
  if (size_update) { ptrd+=std::fread(ptrd,1,size_update,file_update); std::fclose(file_update); }
  if (size_data_gmic_def) { std::memcpy(ptrd,data_gmic_def,size_data_gmic_def-1); ptrd+=size_data_gmic_def-1; }
  *ptrd = 0;

  // Parse filters definitions and create corresponding TreeView store.
  GtkTreeIter iter, parent[16];
  treeview_store = gtk_tree_store_new(2,G_TYPE_UINT,G_TYPE_STRING);
  char preview_command[4096] = { 0 }, arguments[4096] = { 0 }, line[256*1024] = { 0 }, entry[4096] = { 0 }, command[4096] = { 0 };
  char format_entry1[1024] = { 0 }, format_cont1[1024] = { 0 }, format_entry2[1024] = { 0 }, format_cont2[1024] = { 0 };
  const char *const locale = get_locale();
  int level = 0;

  std::sprintf(line,"#@gimp_%s",locale);
  if (std::strstr(gmic_macros,line)) {  // If filters are translated.
    std::sprintf(format_entry1,"#@gimp_%s %%4095[^:]: %%4095[^, ]%%*c %%4095[^, ]%%*c %%4095[^\n]",locale);
    std::sprintf(format_cont1,"#@gimp_%s : %%4095[^\n]",locale);
  } else { // else , use English language for filters.
    std::sprintf(format_entry1,"#@gimp_en %%4095[^:]: %%4095[^, ]%%*c %%4095[^, ]%%*c %%4095[^\n]");
    std::sprintf(format_cont1,"#@gimp_en : %%4095[^\n]");
  }
  std::sprintf(format_entry2,"#@gimp%%*[ ]%%4095[^:]: %%4095[^, ]%%*c %%4095[^, ]%%*c %%4095[^\n]");
  std::sprintf(format_cont2,"#@gimp%%*[ ]: %%4095[^\n]");

  for (const char *data = gmic_macros; *data; ) {
    if (*data=='\n') ++data;
    else {
      if (std::sscanf(data,"%262143[^\n]\n",line)>0) data += std::strlen(line) + 1;
      arguments[0] = 0;
      if (line[0]=='#') {
        int err = std::sscanf(line,format_entry1,entry,command,preview_command,arguments);
        if (err<1) err = std::sscanf(line,format_entry2,entry,command,preview_command,arguments);
        if (err==1) { // If entry is a menu folder.
          cimg::strclean(entry);
          char *nentry = entry;
          while (*nentry=='_') { ++nentry; --level; }
          if (level<0) level = 0;
          if (level>15) level = 15;
          cimg::strclean(nentry);
          if (*nentry) {
            gtk_tree_store_append(treeview_store,&parent[level],level?&parent[level-1]:0);
            gtk_tree_store_set(treeview_store,&parent[level],0,0,1,nentry,-1);
            if (!level) {
              const char *treepath = gtk_tree_model_get_string_from_iter(GTK_TREE_MODEL(treeview_store),&parent[level]);
              gmic_1stlevel_entries.insert(CImg<char>(treepath,std::strlen(treepath)+1,1,1,1,true));
            }
            ++level;
          }
        } else if (err>=2) { // If entry is a regular filter.
          cimg::strclean(entry);
          cimg::strclean(command);
          gmic_entries.insert(CImg<char>(entry,std::strlen(entry)+1));
          gmic_commands.insert(CImg<char>(command,std::strlen(command)+1));
          gmic_arguments.insert(CImg<char>(arguments,std::strlen(arguments)+1));
          if (err>=3) {
            cimg::strclean(preview_command);
            gmic_preview_commands.insert(CImg<char>(preview_command,std::strlen(preview_command)+1));
          } else gmic_preview_commands.insert(1);
          gtk_tree_store_append(treeview_store,&iter,level?&parent[level-1]:0);
          gtk_tree_store_set(treeview_store,&iter,0,gmic_entries.size,1,entry,-1);
        } else if (err==0 && (std::sscanf(line,format_cont1,arguments)==1 ||
                              std::sscanf(line,format_cont2,arguments)==1)) { // Line is an entry continuation
          if (gmic_arguments) {
            gmic_arguments.last().last() = ' ';
            gmic_arguments.last().append(CImg<char>(arguments,std::strlen(arguments)+1),'x');
          }
        }
      }
    }
  }
  return network_update?network_succeed:true;
}

// 'Convert' a CImg<float> image to a CImg<unsigned char> image, withing the same buffer.
//---------------------------------------------------------------------------------------
void convert_image_float2uchar(CImg<float>& img) {
  const unsigned int siz = img.width*img.height;
  unsigned char *ptrd = (unsigned char*)img.ptr();
  switch (img.dim) {
  case 1 : {
    const float *ptr0 = img.ptr(0,0,0,0);
    for (unsigned int i = 0; i<siz; ++i) *(ptrd++) = (unsigned char)*(ptr0++);
  } break;
  case 2 : {
    const float *ptr0 = img.ptr(0,0,0,0), *ptr1 = img.ptr(0,0,0,1);
    for (unsigned int i = 0; i<siz; ++i) {
      *(ptrd++) = (unsigned char)*(ptr0++);
      *(ptrd++) = (unsigned char)*(ptr1++);
    }
  } break;
  case 3 : {
    const float *ptr0 = img.ptr(0,0,0,0), *ptr1 = img.ptr(0,0,0,1), *ptr2 = img.ptr(0,0,0,2);
    for (unsigned int i = 0; i<siz; ++i) {
      *(ptrd++) = (unsigned char)*(ptr0++);
      *(ptrd++) = (unsigned char)*(ptr1++);
      *(ptrd++) = (unsigned char)*(ptr2++);
    }
  } break;
  case 4 : {
    const float *ptr0 = img.ptr(0,0,0,0), *ptr1 = img.ptr(0,0,0,1), *ptr2 = img.ptr(0,0,0,2), *ptr3 = img.ptr(0,0,0,3);
    for (unsigned int i = 0; i<siz; ++i) {
      *(ptrd++) = (unsigned char)*(ptr0++);
      *(ptrd++) = (unsigned char)*(ptr1++);
      *(ptrd++) = (unsigned char)*(ptr2++);
      *(ptrd++) = (unsigned char)*(ptr3++);
    }
  } break;
  default: return;
  }
}

// Calibrate any image to fit the number of required channels (GRAY,GRAYA, RGB or RGBA).
//---------------------------------------------------------------------------------------
void calibrate_image(CImg<float>& img, const unsigned int channels, const bool preview) {
  if (!img || !channels) return;
  switch (channels) {

  case 1 : // To GRAY
    switch (img.dimv()) {
    case 1 : // from GRAY
      break;
    case 2 : // from GRAYA
      if (preview) {
        float *ptr_r = img.ptr(0,0,0,0), *ptr_a = img.ptr(0,0,0,1);
        cimg_forXY(img,x,y) {
          const unsigned int a = (unsigned int)*(ptr_a++), i = 96 + (((x^y)&8)<<3);
          *ptr_r = (float)((a*(unsigned int)*ptr_r + (255-a)*i)>>8);
          ++ptr_r;
        }
      }
      img.channel(0);
      break;
    case 3 : // from RGB
      img.RGBtoYCbCr().channel(0);
      break;
    case 4 : // from RGBA
      img.get_shared_channels(0,2).RGBtoYCbCr();
      if (preview) {
        float *ptr_r = img.ptr(0,0,0,0), *ptr_a = img.ptr(0,0,0,3);
        cimg_forXY(img,x,y) {
          const unsigned int a = (unsigned int)*(ptr_a++), i = 96 + (((x^y)&8)<<3);
          *ptr_r = (float)((a*(unsigned int)*ptr_r + (255-a)*i)>>8);
          ++ptr_r;
        }
      }
      img.channel(0);
      break;
    default : // from multi-channel (>4)
      img.channel(0);
    } break;

  case 2: // To GRAYA
    switch (img.dimv()) {
    case 1: // from GRAY
      img.resize(-100,-100,1,2,0).get_shared_channel(1).fill(255);
      break;
    case 2: // from GRAYA
      break;
    case 3: // from RGB
      img.RGBtoYCbCr().channels(0,1).get_shared_channel(1).fill(255);
      break;
    case 4: // from RGBA
      img.get_shared_channels(0,2).RGBtoYCbCr();
      img.get_shared_channel(1) = img.get_shared_channel(3);
      img.channels(0,1);
      break;
    default: // from multi-channel (>4)
      img.channels(0,1).get_shared_channel(1).fill(255);
    } break;

  case 3: // to RGB
    switch (img.dimv()) {
    case 1: // from GRAY
      img.resize(-100,-100,1,3);
      break;
    case 2: // from GRAYA
      if (preview) {
        float *ptr_r = img.ptr(0,0,0,0), *ptr_a = img.ptr(0,0,0,1);
        cimg_forXY(img,x,y) {
          const unsigned int a = (unsigned int)*(ptr_a++), i = 96 + (((x^y)&8)<<3);
          *ptr_r = (float)((a*(unsigned int)*ptr_r + (255-a)*i)>>8);
          ++ptr_r;
        }
      }
      img.channel(0).resize(-100,-100,1,3);
      break;
    case 3: // from RGB
      break;
    case 4: // from RGBA
      if (preview) {
        float *ptr_r = img.ptr(0,0,0,0), *ptr_g = img.ptr(0,0,0,1), *ptr_b = img.ptr(0,0,0,2), *ptr_a = img.ptr(0,0,0,3);
        cimg_forXY(img,x,y) {
          const unsigned int a = (unsigned int)*(ptr_a++), i = 96 + (((x^y)&8)<<3);
          *ptr_r = (float)((a*(unsigned int)*ptr_r + (255-a)*i)>>8);
          *ptr_g = (float)((a*(unsigned int)*ptr_g + (255-a)*i)>>8);
          *ptr_b = (float)((a*(unsigned int)*ptr_b + (255-a)*i)>>8);
          ++ptr_r; ++ptr_g; ++ptr_b;
        }
      }
      img.channels(0,2);
      break;
    default: // from multi-channel (>4)
      img.channels(0,2);
    } break;

  case 4: // to RGBA
    switch (img.dimv()) {
    case 1: // from GRAY
      img.resize(-100,-100,1,4).get_shared_channel(3).fill(255);
      break;
    case 2: // from GRAYA
      img.resize(-100,-100,1,4,0);
      img.get_shared_channel(3) = img.get_shared_channel(1);
      img.get_shared_channel(1) = img.get_shared_channel(0);
      img.get_shared_channel(2) = img.get_shared_channel(0);
      break;
    case 3: // from RGB
      img.resize(-100,-100,1,4,0).get_shared_channel(3).fill(255);
      break;
    case 4: // from RGBA
      break;
    default: // from multi-channel (>4)
      img.resize(-100,-100,1,4,0);
    } break;
  }
}

// Get the input layers of a GIMP image as a list of CImg<float>.
//---------------------------------------------------------------
template<typename T>
CImg<int> get_input_layers(CImgList<T>& images) {

  // Retrieve the list of desired layers.
  int
    nb_layers = 0,
    *layers = gimp_image_get_layers(image_id,&nb_layers),
    active_layer = gimp_image_get_active_layer(image_id);
  CImg<int> input_layers;
  const unsigned int input_mode = get_input_mode();
  switch (input_mode) {
  case 0 : // Input none
    break;
  case 1 : // Input active layer
    input_layers = CImg<int>::vector(active_layer);
    break;
  case 2 : case 9 : // Input all image layers
    input_layers = CImg<int>(layers,1,nb_layers);
    if (input_mode==9) input_layers.mirror('y');
    break;
  case 3 : { // Input active & below layers
    int i = 0; for (i = 0; i<nb_layers; ++i) if (layers[i]==active_layer) break;
    if (i<nb_layers-1) input_layers = CImg<int>::vector(active_layer,layers[i+1]);
    else input_layers = CImg<int>::vector(active_layer);
  } break;
  case 4 : { // Input active & above layers
    int i = 0; for (i = 0; i<nb_layers; ++i) if (layers[i]==active_layer) break;
    if (i>0) input_layers = CImg<int>::vector(active_layer,layers[i-1]);
    else input_layers = CImg<int>::vector(active_layer);
  } break;
  case 5 : case 7 : { // Input all visible image layers
    CImgList<int> visible_layers;
    for (int i = 0; i<nb_layers; ++i)
      if (gimp_drawable_get_visible(layers[i])) visible_layers.insert(CImg<int>::vector(layers[i]));
    input_layers = visible_layers.get_append('y');
    if (input_mode==7) input_layers.mirror('y');
  } break;
  default : { // Input all invisible image layers
    CImgList<int> invisible_layers;
    for (int i = 0; i<nb_layers; ++i)
      if (!gimp_drawable_get_visible(layers[i])) invisible_layers.insert(CImg<int>::vector(layers[i]));
    input_layers = invisible_layers.get_append('y');
    if (input_mode==8) input_layers.mirror('y');
  } break;
  }

  // Read input image data into a CImgList<float>.
  images.assign(input_layers.height);
  GimpPixelRgn region;
  gint x1, y1, x2, y2;
  cimglist_for(images,l) {
    GimpDrawable *drawable = gimp_drawable_get(input_layers[l]);
    gimp_drawable_mask_bounds(drawable->drawable_id,&x1,&y1,&x2,&y2);
    const int channels = drawable->bpp;
    gimp_pixel_rgn_init(&region,drawable,x1,y1,x2-x1,y2-y1,false,false);
    guchar *const row = g_new(guchar,(x2-x1)*channels), *ptrs = 0;
    CImg<T> img(x2-x1,y2-y1,1,channels);
    switch (channels) {
    case 1 : {
      T *ptr_r = img.ptr(0,0,0,0);
      cimg_forY(img,y) {
        gimp_pixel_rgn_get_row(&region,ptrs=row,x1,y1+y,img.width);
        cimg_forX(img,x) *(ptr_r++) = (T)*(ptrs++);
      }
    } break;
    case 2 : {
      T *ptr_r = img.ptr(0,0,0,0), *ptr_g = img.ptr(0,0,0,1);
      cimg_forY(img,y) {
        gimp_pixel_rgn_get_row(&region,ptrs=row,x1,y1+y,img.width);
        cimg_forX(img,x) { *(ptr_r++) = (T)*(ptrs++); *(ptr_g++) = (T)*(ptrs++); }
      }
    } break;
    case 3 : {
      T *ptr_r = img.ptr(0,0,0,0), *ptr_g = img.ptr(0,0,0,1), *ptr_b = img.ptr(0,0,0,2);
      cimg_forY(img,y) {
        gimp_pixel_rgn_get_row(&region,ptrs=row,x1,y1+y,img.width);
        cimg_forX(img,x) { *(ptr_r++) = (T)*(ptrs++); *(ptr_g++) = (T)*(ptrs++); *(ptr_b++) = (T)*(ptrs++); }
      }
    } break;
    case 4 : {
      T *ptr_r = img.ptr(0,0,0,0), *ptr_g = img.ptr(0,0,0,1), *ptr_b = img.ptr(0,0,0,2), *ptr_a = img.ptr(0,0,0,3);
      cimg_forY(img,y) {
        gimp_pixel_rgn_get_row(&region,ptrs=row,x1,y1+y,img.width);
        cimg_forX(img,x) {
          *(ptr_r++) = (T)*(ptrs++); *(ptr_g++) = (T)*(ptrs++); *(ptr_b++) = (T)*(ptrs++); *(ptr_a++) = (T)*(ptrs++);
        }
      }
    } break;
    }
    g_free(row);
    gimp_drawable_detach(drawable);
    img.transfer_to(images[l]);
  }
  return input_layers;
}

// Return the G'MIC command line needed to run the selected filter.
//-----------------------------------------------------------------
const char* get_commandline(const bool preview) {
  const unsigned int
    filter = get_current_filter(),
    nbparams = get_filter_nbparams(filter);
  if (!filter) return 0;
  static CImg<char> res;
  CImgList<char> lres;
  switch (get_verbosity_mode()) {
  case 0: lres.insert(CImg<char>("-v- -",5)); break;
  case 1: lres.insert(CImg<char>("-",1)); break;
  case 2: lres.insert(CImg<char>("-v+ -v+ -v+ -v+ -v+ -",21)); break;
  default: lres.insert(CImg<char>("-v+ -debug -",12));
  }
  const unsigned int N = filter - 1;
  const CImg<char> &command_item = (preview?gmic_preview_commands[N]:gmic_commands[N]);
  if (command_item) {
    lres.insert(command_item);
    if (nbparams) {
      lres[1].last() = ' ';
      for (unsigned int p = 0; p<nbparams; ++p) {
        const char *const param = get_filter_parameter(filter,p);
        lres.insert(CImg<char>(param,std::strlen(param)+1)).last().last() = ',';
      }
    }
    (res = lres.get_append('x')).last() = 0;
  }
  return res.ptr();
}

// Handle GUI event functions.
//----------------------------

// Handle widgets events related to parameter changes.
void on_float_parameter_changed(GtkAdjustment *scale, gpointer user_data) {
  const unsigned int arg = *(unsigned int*)user_data;
  double value = 0;
  gimp_double_adjustment_update(scale,&value);
  char s_value[1024] = { 0 };
  std::sprintf(s_value,"%g",value);
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_int_parameter_changed(GtkAdjustment *scale, gpointer user_data) {
  const unsigned int arg = *(unsigned int*)user_data;
  int value = 0;
  gimp_int_adjustment_update(scale,&value);
  char s_value[1024] = { 0 };
  std::sprintf(s_value,"%d",value);
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_bool_parameter_changed(GtkCheckButton *checkbutton, gpointer user_data) {
  const unsigned int arg = *(unsigned int*)user_data;
  int value = 0;
  g_object_get(checkbutton,"active",&value,NULL);
  char s_value[1024] = { 0 };
  std::sprintf(s_value,"%d",value?1:0);
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_list_parameter_changed(GtkComboBox *combobox, gpointer user_data) {
  const unsigned int arg = *(unsigned int*)user_data;
  int value = 0;
  g_object_get(combobox,"active",&value,NULL);
  char s_value[1024] = { 0 };
  std::sprintf(s_value,"%d",value);
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_text_parameter_changed(GtkButton *button, gpointer user_data) {
  button = 0;
  const unsigned int
    arg0 = *(unsigned int*)user_data,
    arg = arg0&32767,
    keep_dquote = arg0&32768;
  GtkWidget *entry = *((GtkWidget**)user_data+1);
  const char *s_value = gtk_entry_get_text(GTK_ENTRY(entry));
  if (keep_dquote) {
    char tmp[1024] = { 0 };
    std::sprintf(tmp,"\"%s\"",s_value);
    set_filter_parameter(get_current_filter(),arg,tmp);
  } else set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_file_parameter_changed(GtkFileChooserButton *widget, gpointer user_data){
  const unsigned int arg = *(unsigned int*)user_data;
  const char
    *const filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(widget)),
    *s_value = filename?filename:"";
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

void on_color_parameter_changed(GtkColorButton *widget, gpointer user_data){
  const unsigned int arg = *(unsigned int*)user_data;
  GdkColor color;
  gtk_color_button_get_color(GTK_COLOR_BUTTON(widget),&color);
  char s_value[1024] = { 0 };
  if (gtk_color_button_get_use_alpha(GTK_COLOR_BUTTON(widget)))
    std::sprintf(s_value,"%d,%d,%d,%d",
                 color.red>>8,color.green>>8,color.blue>>8,gtk_color_button_get_alpha(GTK_COLOR_BUTTON(widget))>>8);
  else std::sprintf(s_value,"%d,%d,%d",
                    color.red>>8,color.green>>8,color.blue>>8);
  set_filter_parameter(get_current_filter(),arg,s_value);
  return_create_dialog = true;
}

// Handle responses to the dialog window buttons.
void create_parameters_gui(const bool);
void process_image(const char *);
void process_preview();

void on_dialog_input_mode_changed(GtkComboBox *combobox, gpointer user_data) {
  user_data = 0;
  int value = 0;
  g_object_get(combobox,"active",&value,NULL);
  if (value<2) gtk_combo_box_set_active(combobox,value=3);
  set_input_mode((unsigned int)value);
}

void on_dialog_preview_mode_changed(GtkComboBox *combobox, gpointer user_data) {
  user_data = 0;
  int value = 0;
  g_object_get(combobox,"active",&value,NULL);
  if (value<2) gtk_combo_box_set_active(combobox,value=2);
  set_preview_mode((unsigned int)value);
}

void on_dialog_output_mode_changed(GtkComboBox *combobox, gpointer user_data) {
  user_data = 0;
  int value = 0;
  g_object_get(combobox,"active",&value,NULL);
  if (value<2) gtk_combo_box_set_active(combobox,value=2);
  set_output_mode((unsigned int)value);
}

void on_dialog_verbosity_mode_changed(GtkComboBox *combobox, gpointer user_data) {
  user_data = 0;
  int value = 0;
  g_object_get(combobox,"active",&value,NULL);
  if (value<2) gtk_combo_box_set_active(combobox,value=2);
  set_verbosity_mode((unsigned int)value);
}

void on_dialog_reset_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0; user_data = 0;
  create_parameters_gui(true);
  return_create_dialog = true;
}

void on_dialog_cancel_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0; user_data = 0;
  return_create_dialog = false;
  gtk_main_quit();
}

void on_dialog_apply_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0; user_data = 0;
  process_image(0);
  return_create_dialog = false;
}

void on_dialog_ok_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0; user_data = 0;
  gtk_main_quit();
}

void on_dialog_net_update_toggled(GtkCheckButton *widget, gpointer user_data) {
  widget = 0;
  const bool net_update = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(user_data));
  set_net_update(net_update);
}

void on_dialog_treemode_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0;
  GtkWidget *treeview = GTK_WIDGET(user_data);
  set_treemode(!get_treemode());
  flush_treeview(treeview);
}

void on_dialog_update_clicked(GtkButton *widget, gpointer user_data) {
  widget = 0;
  if (!update_filters_definition(get_net_update())) {
    GtkWidget *dialog = 0;
    gimp_get_data("gmic_gui_dialog",&dialog);
    GtkWidget *message = gtk_message_dialog_new_with_markup(dialog?GTK_WINDOW(dialog):0,GTK_DIALOG_MODAL,GTK_MESSAGE_ERROR,GTK_BUTTONS_OK,
                                                            translate(0),gmic_update_server,gmic_update_file,gmic_update_file);
    gtk_widget_show(message);
    gtk_dialog_run(GTK_DIALOG(message));
    gtk_widget_destroy(message);
  } else reset_filters_parameters();
  const unsigned int filter = get_current_filter();
  GtkWidget *treeview = GTK_WIDGET(user_data);
  gtk_tree_view_set_model(GTK_TREE_VIEW(treeview),GTK_TREE_MODEL(treeview_store));
  set_current_filter(filter);
  flush_treeview(treeview);
  create_parameters_gui(true);
}

// Event occuring when selected filter has changed.
void on_filter_changed(GtkTreeSelection *selection, gpointer user_data) {
  user_data = 0;
  GtkTreeIter iter;
  GtkTreeModel *model;
  unsigned int choice = 0;
  if (gtk_tree_selection_get_selected(selection,&model,&iter)) {
    gtk_tree_model_get(model,&iter,0,&choice,-1);
    const char *const treepath = gtk_tree_model_get_string_from_iter(GTK_TREE_MODEL(treeview_store),&iter);
    gimp_set_data("gmic_current_treepath",treepath,std::strlen(treepath)+1);
  }
  set_current_filter(choice);
  create_parameters_gui(false);
  return_create_dialog = true;
}

// Event occuring when image preview need to be updated.
void on_preview_invalidated(GimpPreview *preview) {
  preview = 0;
  if (gimp_image_is_valid(image_id)) {
    if (!gimp_drawable_is_valid(drawable_preview->drawable_id)) {
      gtk_widget_destroy(gui_preview);
      drawable_preview = gimp_drawable_get(gimp_image_get_active_drawable(image_id));
      gui_preview = gimp_zoom_preview_new(drawable_preview);
      gtk_widget_show(gui_preview);
      gtk_box_pack_end(GTK_BOX(left_pane),gui_preview,true,true,0);
      g_signal_connect(gui_preview,"invalidated",G_CALLBACK(process_preview),0);
    } else gimp_preview_invalidate((GimpPreview*)gui_preview);
  }
}

// Process image data with the G'MIC interpreter.
//-----------------------------------------------

// This structure stores the arguments required by the processing thread.
struct st_process_thread {
  CImgList<float> images;
  bool is_thread;
  const char *commandline;
  unsigned int verbosity_mode;
#if !defined(__MACOSX__)  && !defined(__APPLE__)
  pthread_mutex_t is_running;
  pthread_t thread;
#endif
};

// Thread that runs the G'MIC interpreter.
void *process_thread(void *arg) {
  st_process_thread &spt = *(st_process_thread*)arg;
  try {
    if (spt.verbosity_mode>0)
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Running G'MIC, with command : '%s'\n",spt.commandline);
    std::setlocale(LC_NUMERIC,"C");
    gmic(spt.commandline,spt.images,gmic_macros);
    if (spt.verbosity_mode>0)
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : G'MIC successfully returned !\n");
  } catch (gmic_exception &e) {
    if (spt.verbosity_mode>0)
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Error encountered when running G'MIC :\n*** %s\n",e.message);
    spt.images.assign();
  }
#if !defined(__MACOSX__)  && !defined(__APPLE__)
  if (spt.is_thread) {
    pthread_mutex_unlock(&spt.is_running);
    pthread_exit(0);
  }
#endif
  return 0;
}

// Process the selected image/layers.
//------------------------------------
void process_image(const char *last_commandline) {
  if (!gimp_image_is_valid(image_id)) return;
  const unsigned int filter = get_current_filter();
  if (!last_commandline && !filter) return;
  const char *commandline = last_commandline?last_commandline:get_commandline(false);
  if (!commandline) return;
  gimp_progress_init_printf(" G'MIC : %s...",gmic_entries[filter-1].ptr());

  // Get input layers for the chosen filter.
  st_process_thread spt;
  spt.is_thread = true;
  spt.commandline = commandline;
  spt.verbosity_mode = get_verbosity_mode();
  const CImg<int> layers = get_input_layers(spt.images);
  CImg<int> dimensions(spt.images.size,4);
  cimglist_for(spt.images,l) {
    const CImg<float>& img = spt.images[l];
    dimensions(l,0) = img.dimx(); dimensions(l,1) = img.dimy();
    dimensions(l,2) = img.dimz(); dimensions(l,3) = img.dimv();
  }

  // Create processing thread and wait for its completion.
#if !defined(__MACOSX__)  && !defined(__APPLE__)
  pthread_mutex_init(&spt.is_running,0);
  pthread_mutex_lock(&spt.is_running);
  pthread_create(&(spt.thread),0,process_thread,(void*)&spt);
  while (pthread_mutex_trylock(&spt.is_running)) { gimp_progress_pulse(); cimg::wait(500); }
  gimp_progress_update(1.0);
  pthread_join(spt.thread,0);
  pthread_mutex_unlock(&spt.is_running);
  pthread_mutex_destroy(&spt.is_running);
#else
  gimp_progress_update(0.5);
  process_thread(&spt);
#endif

  // Get output layers dimensions and check if input/output layers have compatible dimensions.
  unsigned int max_width = 0, max_height = 0, max_channels = 0;
  cimglist_for(spt.images,l) {
    if (spt.images[l].width>max_width)   max_width = spt.images[l].width;
    if (spt.images[l].height>max_height) max_height = spt.images[l].height;
    if (spt.images[l].dim>max_channels)  max_channels = spt.images[l].dim;
  }
  bool is_compatible_dimensions = (spt.images.size==layers.height);
  for (unsigned int p = 0; p<spt.images.size && is_compatible_dimensions; ++p) {
    const CImg<float>& img = spt.images[p];
    if (img.dimx()!=dimensions(p,0) ||
        img.dimy()!=dimensions(p,1) ||
        img.dimv()>dimensions(p,3)) is_compatible_dimensions = false;
  }

  // Transfer the output layers back into GIMP.
  GimpPixelRgn region;
  gint x1, y1, x2, y2;
  switch (get_output_mode()) {
  case 0 : { // Output in 'Replace' mode.
    gimp_image_undo_group_start(image_id);
    if (is_compatible_dimensions) cimglist_for(spt.images,l) { // Direct replacement of the layer data.
      CImg<float> &img = spt.images[l];
      calibrate_image(img,dimensions(l,3),false);
      GimpDrawable *drawable = gimp_drawable_get(layers[l]);
      gimp_drawable_mask_bounds(drawable->drawable_id,&x1,&y1,&x2,&y2);
      gimp_pixel_rgn_init(&region,drawable,x1,y1,x2-x1,y2-y1,true,true);
      convert_image_float2uchar(img);
      gimp_pixel_rgn_set_rect(&region,(guchar*)img.ptr(),x1,y1,x2-x1,y2-y1);
      img.assign();
      gimp_drawable_flush(drawable);
      gimp_drawable_merge_shadow(drawable->drawable_id,true);
      gimp_drawable_update(drawable->drawable_id,x1,y1,x2-x1,y2-y1);
      gimp_drawable_detach(drawable);
    } else { // Indirect replacement : create new layers.
      gimp_selection_none(image_id);
      const int layer_pos = gimp_image_get_layer_position(image_id,layers[0]);
      for (unsigned int i = 0; i<layers.height; ++i) gimp_image_remove_layer(image_id,layers[i]);
      cimglist_for(spt.images,p) {
        CImg<float> &img = spt.images[p];
        if (gimp_image_base_type(image_id)==GIMP_GRAY) calibrate_image(img,(img.dimv()==1 || img.dimv()==3)?1:2,false);
        else calibrate_image(img,(img.dimv()==1 || img.dimv()==3)?3:4,false);
        gint layer_id = gimp_layer_new(image_id,"image",img.dimx(),img.dimy(),
                                       img.dimv()==1?GIMP_GRAY_IMAGE:
                                       img.dimv()==2?GIMP_GRAYA_IMAGE:
                                       img.dimv()==3?GIMP_RGB_IMAGE:GIMP_RGBA_IMAGE,
                                       100.0,GIMP_NORMAL_MODE);
        gimp_image_add_layer(image_id,layer_id,layer_pos+p);
        GimpDrawable *drawable = gimp_drawable_get(layer_id);
        gimp_pixel_rgn_init(&region,drawable,0,0,drawable->width,drawable->height,true,true);
        convert_image_float2uchar(img);
        gimp_pixel_rgn_set_rect(&region,(guchar*)img.ptr(),0,0,img.dimx(),img.dimy());
        img.assign();
        gimp_drawable_flush(drawable);
        gimp_drawable_merge_shadow(drawable->drawable_id,true);
        gimp_drawable_update(drawable->drawable_id,0,0,drawable->width,drawable->height);
        gimp_drawable_detach(drawable);
      }
      gimp_image_resize_to_layers(image_id);
    }
    gimp_image_undo_group_end(image_id);
  } break;
  case 1 : { // Output in 'New layer(s)' mode.
    gimp_image_undo_group_start(image_id);
    gimp_selection_none(image_id);
    cimglist_for(spt.images,p) {
      CImg<float> &img = spt.images[p];
      if (gimp_image_base_type(image_id)==GIMP_GRAY)
        calibrate_image(img,(img.dimv()==1 || img.dimv()==3)?1:2,false);
      else
        calibrate_image(img,(img.dimv()==1 || img.dimv()==3)?3:4,false);
      gint layer_id = gimp_layer_new(image_id,"image",img.dimx(),img.dimy(),
                                     img.dimv()==1?GIMP_GRAY_IMAGE:
                                     img.dimv()==2?GIMP_GRAYA_IMAGE:
                                     img.dimv()==3?GIMP_RGB_IMAGE:GIMP_RGBA_IMAGE,
                                     100.0,GIMP_NORMAL_MODE);
      gimp_image_add_layer(image_id,layer_id,p);
      GimpDrawable *drawable = gimp_drawable_get(layer_id);
      gimp_pixel_rgn_init(&region,drawable,0,0,drawable->width,drawable->height,true,true);
      convert_image_float2uchar(img);
      gimp_pixel_rgn_set_rect(&region,(guchar*)img.ptr(),0,0,img.dimx(),img.dimy());
      img.assign();
      gimp_drawable_flush(drawable);
      gimp_drawable_merge_shadow(drawable->drawable_id,true);
      gimp_drawable_update(drawable->drawable_id,0,0,drawable->width,drawable->height);
      gimp_drawable_detach(drawable);
    }
    gimp_image_resize_to_layers(image_id);
    gimp_image_undo_group_end(image_id);
  } break;
  default : { // Output in 'New image' mode.
    if (spt.images.size) {
      const int nimage_id = gimp_image_new(max_width,max_height,max_channels<=2?GIMP_GRAY:GIMP_RGB);
      gimp_image_undo_group_start(nimage_id);
      cimglist_for(spt.images,p) {
        CImg<float> &img = spt.images[p];
        if (gimp_image_base_type(nimage_id)!=GIMP_GRAY)
          calibrate_image(img,(img.dimv()==1 || img.dimv()==3)?3:4,false);
        gint layer_id = gimp_layer_new(nimage_id,"image",img.dimx(),img.dimy(),
                                       img.dimv()==1?GIMP_GRAY_IMAGE:
                                       img.dimv()==2?GIMP_GRAYA_IMAGE:
                                       img.dimv()==3?GIMP_RGB_IMAGE:GIMP_RGBA_IMAGE,
                                       100.0,GIMP_NORMAL_MODE);
        gimp_image_add_layer(nimage_id,layer_id,p);
        GimpDrawable *drawable = gimp_drawable_get(layer_id);
        GimpPixelRgn dest_region;
        gimp_pixel_rgn_init(&dest_region,drawable,0,0,drawable->width,drawable->height,true,true);
        convert_image_float2uchar(img);
        gimp_pixel_rgn_set_rect(&dest_region,(guchar*)img.ptr(),0,0,img.dimx(),img.dimy());
        img.assign();
        gimp_drawable_flush(drawable);
        gimp_drawable_merge_shadow(drawable->drawable_id,true);
        gimp_drawable_update(drawable->drawable_id,0,0,drawable->width,drawable->height);
        gimp_drawable_detach(drawable);
      }
      gimp_display_new(nimage_id);
      gimp_image_undo_group_end(nimage_id);
    }
  }
  }
  gimp_progress_end();
  gimp_displays_flush();
}

// Process the preview image.
//---------------------------
void process_preview() {
  if (!gimp_image_is_valid(image_id)) return;
  const unsigned int filter = get_current_filter();
  if (!filter) return;
  const char *const commandline = get_commandline(true);
  if (!commandline) return;

  // Get input layers for the chosen filter and convert then to the preview size if necessary.
  st_process_thread spt;
  spt.is_thread = false;
  spt.commandline = commandline;
  spt.verbosity_mode = get_verbosity_mode();

  const unsigned int input_mode = get_input_mode();
  int w, h, channels, nb_layers = 0, *layers = gimp_image_get_layers(image_id,&nb_layers);
  guchar *const ptr0 = gimp_zoom_preview_get_source(GIMP_ZOOM_PREVIEW(gui_preview),&w,&h,&channels), *ptrs = ptr0;
  if (nb_layers && input_mode) {
    if (input_mode==1 ||
        (input_mode==2 && nb_layers==1) ||
        (input_mode==3 && nb_layers==1 && gimp_drawable_get_visible(layers[0])) ||
        (input_mode==4 && nb_layers==1 && !gimp_drawable_get_visible(layers[0])) ||
        (input_mode==5 && nb_layers==1)) { // If only one input layer, use the thumbnail provided by GIMP.
      spt.images.assign(1,w,h,1,channels);
      const int wh = w*h;
      switch (channels) {
      case 1 : {
        float *ptr_r = spt.images[0].ptr(0,0,0,0);
        for (int xy = 0; xy<wh; ++xy) *(ptr_r++) = (float)*(ptrs++);
      } break;
      case 2 : {
        float *ptr_r = spt.images[0].ptr(0,0,0,0), *ptr_g = spt.images[0].ptr(0,0,0,1);
        for (int xy = 0; xy<wh; ++xy) { *(ptr_r++) = (float)*(ptrs++); *(ptr_g++) = (float)*(ptrs++);
        }
      } break;
      case 3 : {
        float *ptr_r = spt.images[0].ptr(0,0,0,0), *ptr_g = spt.images[0].ptr(0,0,0,1), *ptr_b = spt.images[0].ptr(0,0,0,2);
        for (int xy = 0; xy<wh; ++xy) {
          *(ptr_r++) = (float)*(ptrs++); *(ptr_g++) = (float)*(ptrs++); *(ptr_b++) = (float)*(ptrs++);
        }
      } break;
      case 4 : {
        float
          *ptr_r = spt.images[0].ptr(0,0,0,0), *ptr_g = spt.images[0].ptr(0,0,0,1),
          *ptr_b = spt.images[0].ptr(0,0,0,2), *ptr_a = spt.images[0].ptr(0,0,0,3);
        for (int xy = 0; xy<wh; ++xy) {
          *(ptr_r++) = (float)*(ptrs++); *(ptr_g++) = (float)*(ptrs++); *(ptr_b++) = (float)*(ptrs++); *(ptr_a++) = (float)*(ptrs++);
        }
      } break;
      }
    } else { // Else, compute a 'hand-made' set of thumbnails.
      CImgList<unsigned char> images_uchar;
      get_input_layers(images_uchar);
      unsigned int wmax = 0, hmax = 0;
      cimglist_for(images_uchar,l) {
        if (images_uchar[l].width>wmax) wmax = images_uchar[l].width;
        if (images_uchar[l].height>hmax) hmax = images_uchar[l].height;
      }
      const double factor = gimp_zoom_preview_get_factor((GimpZoomPreview*)gui_preview);
      int xp, yp;
      gimp_preview_get_position((GimpPreview*)gui_preview,&xp,&yp);
      const int
        x0 = (int)(xp/factor)*wmax/w,
        y0 = (int)(yp/factor)*hmax/h,
        x1 = (int)((xp+w)/factor)*wmax/w - 1,
        y1 = (int)((yp+h)/factor)*hmax/h - 1;
      spt.images.assign(images_uchar.size);
      cimglist_for(images_uchar,l) {
        images_uchar[l].get_crop(x0,y0,x1,y1).resize(w,h,1,-100).transfer_to(spt.images[l]);
        images_uchar[l].assign();
      }
    }
  }

  // Run G'MIC.
  process_thread(&spt);

  // Transfer the output layers back into GIMP preview.
  CImg<float> img;
  switch (get_preview_mode()) {
  case 0 : // Preview 1st layer
    if (spt.images && spt.images.size>0) spt.images[0].transfer_to(img);
    calibrate_image(img,channels,true);
    break;
  case 1 : // Preview 2nd layer
    if (spt.images && spt.images.size>1) spt.images[1].transfer_to(img);
    calibrate_image(img,channels,true);
    break;
  case 2 : // Preview 3rd layer
    if (spt.images && spt.images.size>2) spt.images[2].transfer_to(img);
    calibrate_image(img,channels,true);
    break;
  case 3 : // Preview 4th layer
    if (spt.images && spt.images.size>2) spt.images[3].transfer_to(img);
    calibrate_image(img,channels,true);
    break;
  case 4 : { // Preview 1st->2nd layers
    const unsigned int m = cimg::min(spt.images.size-1,1U);
    CImgList<float> res = spt.images.get_crop(0,m,true);
    cimglist_for(res,l) calibrate_image(res[l],channels,true);
    res.get_append('x').transfer_to(img);
  } break;
  case 5 : { // Preview 1st->3nd layers
    const unsigned int m = cimg::min(spt.images.size-1,2U);
    CImgList<float> res = spt.images.get_crop(0,m,true);
    cimglist_for(res,l) calibrate_image(res[l],channels,true);
    res.get_append('x').transfer_to(img);
  } break;
  case 6 : { // Preview 1st->4nd layers
    const unsigned int m = cimg::min(spt.images.size-1,3U);
    CImgList<float> res = spt.images.get_crop(0,m,true);
    cimglist_for(res,l) calibrate_image(res[l],channels,true);
    res.get_append('x').transfer_to(img);
  } break;
  default : // Preview all layers
    cimglist_for(spt.images,l) calibrate_image(spt.images[l],channels,true);
    spt.images.get_append('x').transfer_to(img);
  }
  spt.images.assign();
  if (!img) { img.assign(w,h,1,4,0); calibrate_image(img,channels,true); }
  if (img.width>img.height) {
    const unsigned int _nh = img.height*w/img.width, nh = _nh?_nh:1;
    img.resize(w,nh,1,-100,2);
  } else {
    const unsigned int _nw = img.width*h/img.height, nw = _nw?_nw:1;
    img.resize(nw,h,1,-100,2);
  }
  if (img.dimx()!=w || img.dimy()!=h) img.resize(w,h,1,-100,0,0,1);
  convert_image_float2uchar(img);
  std::memcpy(ptr0,img.ptr(),w*h*channels*sizeof(unsigned char));
  gimp_preview_draw_buffer((GimpPreview*)gui_preview,ptr0,w*channels);
  g_free(ptr0);
}

// Create the parameters GUI for the chosen filter.
//--------------------------------------------------
void create_parameters_gui(const bool reset_params) {
  const unsigned int filter = get_current_filter();

  // Remove widget in the current frame if necessary.
  GtkWidget *frame = 0;
  gimp_get_data("gmic_gui_frame",&frame);
  if (frame) {
    GtkWidget *child = GTK_WIDGET(gtk_bin_get_child(GTK_BIN(frame)));
    if (child) gtk_container_remove(GTK_CONTAINER(frame),child);
  }

  GtkWidget *table = 0;
  if (!filter) {  // No filter selected -> Default message.
    table = gtk_table_new(1,1,false);
    gtk_widget_show(table);
    GtkWidget *label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(label),translate("<i>Select a filter...</i>"));
    gtk_widget_show(label);
    gtk_table_attach(GTK_TABLE(table),label,0,1,0,1,
                     (GtkAttachOptions)(GTK_EXPAND),(GtkAttachOptions)(GTK_EXPAND),0,0);
    gtk_misc_set_alignment (GTK_MISC(label),0,0.5);
    gtk_frame_set_label(GTK_FRAME(frame),NULL);
  } else { // Filter selected -> Build the parameter table.
    const unsigned int N = filter - 1;
    char nlabel[1024] = { 0 };
    std::sprintf(nlabel,"<b>  %s : </b>",gmic_entries[N].ptr());
    GtkWidget *frame_title = gtk_label_new(NULL);
    gtk_widget_show(frame_title);
    gtk_label_set_markup(GTK_LABEL(frame_title),nlabel);
    gtk_frame_set_label_widget(GTK_FRAME(frame),frame_title);

    char argname[4096] = { 0 }, argtype[4096] = { 0 }, argarg[4096] = { 0 };
    unsigned int nb_arguments = 0;
    for (const char *argument = gmic_arguments[N].ptr(); *argument; ) {
      int err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-z](%4095[^)]",argname,argtype,&(argarg[0]=0));
      if (err!=3) err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-z]{%4095[^}]",argname,argtype,argarg);
      if (err!=3) err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-z][%4095[^]]",argname,argtype,argarg);
      if (err>=2) {
        argument += std::strlen(argname) + std::strlen(argtype) + std::strlen(argarg) + 3;
        if (*argument) ++argument;
        ++nb_arguments;
      } else break;
    }

    if (!nb_arguments) { // Selected filter has no parameters -> Default message.
      table = gtk_table_new(1,1,false);
      gtk_widget_show(table);
      GtkWidget *label = gtk_label_new(NULL);
      gtk_label_set_markup(GTK_LABEL(label),translate("<i>No parameters to set...</i>"));
      gtk_widget_show(label);
      gtk_table_attach(GTK_TABLE(table),label,0,1,0,1,
                       (GtkAttachOptions)(GTK_EXPAND),(GtkAttachOptions)(GTK_EXPAND),0,0);
      gtk_misc_set_alignment(GTK_MISC(label),0,0.5);
    } else { // Selected filter has parameters -> Create parameter table.

      // Create new table for putting parameters inside.
      table = gtk_table_new(3,nb_arguments,false);
      gtk_widget_show(table);
      gtk_table_set_row_spacings(GTK_TABLE(table),6);
      gtk_table_set_col_spacings(GTK_TABLE(table),6);
      gtk_container_set_border_width(GTK_CONTAINER(table),8);

      // Parse arguments list and add recognized one to the table.
      event_infos = new void*[2*nb_arguments];
      int current_parameter = 0, current_line = 0;
      for (const char *argument = gmic_arguments[N].ptr(); *argument; ) {
        int err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-Z](%4095[^)]",argname,argtype,&(argarg[0]=0));
        if (err!=3) err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-Z][%4095[^]]",argname,argtype,argarg);
        if (err!=3) err = std::sscanf(argument,"%4095[^=]=%4095[ a-zA-Z]{%4095[^}]",argname,argtype,argarg);
        if (err>=2) {
          argument += std::strlen(argname) + std::strlen(argtype) + std::strlen(argarg) + 3;
          if (*argument) ++argument;
          cimg::strclean(argname);
          cimg::strescape(argname);
          cimg::strclean(argtype);
          const char *const s_value = get_filter_parameter(filter,current_parameter);

          // Check for a float-valued parameter -> Create GimpScaleEntry.
          bool found_valid_item = false;
          if (!found_valid_item && !cimg::strcasecmp(argtype,"float")) {
            float initial_value = 0, min_value = 0, max_value = 100;
            std::setlocale(LC_NUMERIC,"C");
            std::sscanf(argarg,"%f%*c%f%*c%f",&initial_value,&min_value,&max_value);
            if (!reset_params && std::sscanf(s_value,"%f",&initial_value)) {}
            GtkObject *scale = gimp_scale_entry_new(GTK_TABLE(table),0,current_line,argname,100,6,
                                                    (gdouble)initial_value,(gdouble)min_value,(gdouble)max_value,
                                                    0.1,0.1,2,true,0,0,0,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            on_float_parameter_changed(GTK_ADJUSTMENT(scale),(void*)(event_infos+2*current_parameter));
            g_signal_connect(scale,"value_changed",G_CALLBACK(on_float_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(scale,"value_changed",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for an int-valued parameter -> Create GimpScaleEntry.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"int")) {
            float initial_value = 0, min_value = 0, max_value = 100;
            std::setlocale(LC_NUMERIC,"C");
            std::sscanf(argarg,"%f%*c%f%*c%f",&initial_value,&min_value,&max_value);
            if (!reset_params && std::sscanf(s_value,"%f",&initial_value)) {}
            GtkObject *scale = gimp_scale_entry_new(GTK_TABLE(table),0,current_line,argname,100,6,
                                                    (gdouble)(int)initial_value,(gdouble)(int)min_value,
                                                    (gdouble)(int)max_value,
                                                    1,1,0,true,0,0,0,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            on_int_parameter_changed(GTK_ADJUSTMENT(scale),(void*)(event_infos+2*current_parameter));
            g_signal_connect(scale,"value_changed",G_CALLBACK(on_int_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(scale,"value_changed",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a bool-valued parameter -> Create GtkCheckButton.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"bool")) {
            const bool
              initial_value = (!cimg::strcasecmp(argarg,"true") || (argarg[0]=='1' && argarg[1]==0)),
              current_value = (!cimg::strcasecmp(s_value,"true") || (s_value[0]=='1' && s_value[1]==0)),
              state = (reset_params || !*s_value)?initial_value:current_value;
            GtkWidget *checkbutton = gtk_check_button_new_with_label(argname);
            gtk_widget_show(checkbutton);
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(checkbutton),state?true:false);
            gtk_table_attach(GTK_TABLE(table),checkbutton,0,2,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            on_bool_parameter_changed(GTK_CHECK_BUTTON(checkbutton),(void*)(event_infos+2*current_parameter));
            g_signal_connect(checkbutton,"toggled",G_CALLBACK(on_bool_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(checkbutton,"toggled",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a list-valued parameter -> Create GtkComboBox.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"choice")) {
            GtkWidget *label = gtk_label_new(argname);
            gtk_widget_show(label);
            gtk_table_attach(GTK_TABLE(table),label,0,1,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            gtk_misc_set_alignment(GTK_MISC(label),0,0.5);
            GtkWidget *combobox = gtk_combo_box_new_text();
            gtk_widget_show(combobox);
            char s_entry[4096] = { 0 }, end = 0; int err2 = 0;
            unsigned int initial_value = 0;
            const char *entries = argarg;
            if (std::sscanf(entries,"%u",&initial_value)==1) entries+=std::sprintf(s_entry,"%u",initial_value) + 1;
            while (*entries) {
              if ((err2 = std::sscanf(entries,"%4095[^,]%c",s_entry,&end))>0) {
                entries += std::strlen(s_entry) + (err2==2?1:0);
                cimg::strclean(s_entry);
                gtk_combo_box_append_text(GTK_COMBO_BOX(combobox),s_entry);
              } else break;
            }
            if (!reset_params && std::sscanf(s_value,"%u",&initial_value)) {}
            gtk_combo_box_set_active(GTK_COMBO_BOX(combobox),initial_value);
            gtk_table_attach(GTK_TABLE(table),combobox,1,3,current_line,current_line+1,
                             (GtkAttachOptions)(GTK_EXPAND | GTK_FILL),(GtkAttachOptions)(GTK_FILL),0,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            on_list_parameter_changed(GTK_COMBO_BOX(combobox),(void*)(event_infos+2*current_parameter));
            g_signal_connect(combobox,"changed",G_CALLBACK(on_list_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(combobox,"changed",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a text-valued parameter -> Create GtkEntry.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"text")) {
            GtkWidget *label = gtk_label_new(argname);
            gtk_widget_show(label);
            gtk_table_attach(GTK_TABLE(table),label,0,1,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            gtk_misc_set_alignment(GTK_MISC(label),0,0.5);
            GtkWidget *entry = gtk_entry_new_with_max_length(4095);
            gtk_widget_show(entry);
            const bool keep_dquote = (argarg[0]=='"' && argarg[std::strlen(argarg)-1]=='"');
            cimg::strclean(argarg);
            if (!reset_params && *s_value) {
              char tmp[1024] = { 0 };
              std::strcpy(tmp,s_value);
              cimg::strclean(tmp);
              gtk_entry_set_text(GTK_ENTRY(entry),tmp);
            } else gtk_entry_set_text(GTK_ENTRY(entry),argarg);
            gtk_table_attach(GTK_TABLE(table),entry,1,2,current_line,current_line+1,
                             (GtkAttachOptions)(GTK_EXPAND | GTK_FILL),(GtkAttachOptions)0,0,0);
            GtkWidget *button = gtk_button_new_with_label(translate("Update"));
            gtk_widget_show(button);
            gtk_table_attach(GTK_TABLE(table),button,2,3,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            event_infos[2*current_parameter] = (void*)(current_parameter | (keep_dquote?32768:0));
            event_infos[2*current_parameter+1] = (void*)entry;
            on_text_parameter_changed(GTK_BUTTON(button),(void*)(event_infos+2*current_parameter));
            g_signal_connect(button,"clicked",G_CALLBACK(on_text_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect(entry,"changed",G_CALLBACK(on_text_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(button,"clicked",G_CALLBACK(on_preview_invalidated),0);
            g_signal_connect_swapped(entry,"activate",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a filename parameter -> Create GtkFileChooserButton.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"file")) {
            GtkWidget *label = gtk_label_new(argname);
            gtk_widget_show(label);
            gtk_table_attach(GTK_TABLE(table),label,0,1,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            gtk_misc_set_alignment(GTK_MISC(label),0,0.5);
            GtkWidget *filechooser = gtk_file_chooser_button_new(argname,GTK_FILE_CHOOSER_ACTION_OPEN);
            gtk_widget_show(filechooser);
            cimg::strclean(argarg);
            if (!reset_params && *s_value) gtk_file_chooser_set_filename(GTK_FILE_CHOOSER(filechooser),s_value);
            else gtk_file_chooser_set_filename(GTK_FILE_CHOOSER(filechooser),argarg);
            gtk_table_attach(GTK_TABLE(table),filechooser,1,3,current_line,current_line+1,
                             (GtkAttachOptions)(GTK_EXPAND | GTK_FILL),(GtkAttachOptions)0,0,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            on_file_parameter_changed(GTK_FILE_CHOOSER_BUTTON(filechooser),(void*)(event_infos+2*current_parameter));
            g_signal_connect(filechooser,"file-set",G_CALLBACK(on_file_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(filechooser,"file-set",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a color -> Create GtkColorButton.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"color")) {
            GtkWidget *hbox = gtk_hbox_new(false,6);
            gtk_widget_show(hbox);
            gtk_table_attach(GTK_TABLE(table),hbox,0,2,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            GtkWidget *label = gtk_label_new(argname);
            gtk_widget_show(label);
            gtk_box_pack_start(GTK_BOX(hbox),label,false,false,0);
            GtkWidget *colorchooser = gtk_color_button_new();
            gtk_widget_show(colorchooser);
            gtk_color_button_set_title(GTK_COLOR_BUTTON(colorchooser),argname);
            gtk_box_pack_start(GTK_BOX(hbox),colorchooser,false,false,0);
            event_infos[2*current_parameter] = (void*)current_parameter;
            event_infos[2*current_parameter+1] = (void*)0;
            cimg::strclean(argarg);
            unsigned int red = 0, green = 0, blue = 0, alpha = 255;
            const int err = std::sscanf(argarg,"%u%*c%u%*c%u%*c%u",&red,&green,&blue,&alpha);
            if (!reset_params && std::sscanf(s_value,"%u%*c%u%*c%u%*c%u",&red,&green,&blue,&alpha)==err) {}
            GdkColor col;
            col.pixel = 0; col.red = red<<8; col.green = green<<8; col.blue = blue<<8;
            gtk_color_button_set_color(GTK_COLOR_BUTTON(colorchooser),&col);
            if (err==4) {
              gtk_color_button_set_use_alpha(GTK_COLOR_BUTTON(colorchooser),true);
              gtk_color_button_set_alpha(GTK_COLOR_BUTTON(colorchooser),alpha<<8);
            } else gtk_color_button_set_use_alpha(GTK_COLOR_BUTTON(colorchooser),false);
            on_color_parameter_changed(GTK_COLOR_BUTTON(colorchooser),(void*)(event_infos+2*current_parameter));
            g_signal_connect(colorchooser,"color-set",G_CALLBACK(on_color_parameter_changed),
                             (void*)(event_infos+2*current_parameter));
            g_signal_connect_swapped(colorchooser,"color-set",G_CALLBACK(on_preview_invalidated),0);
            found_valid_item = true;
            ++current_parameter;
          }

          // Check for a note -> Create GtkLabel.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"note")) {
            cimg::strclean(argarg);
            cimg::strescape(argarg);
            GtkWidget *label = gtk_label_new(NULL);
            gtk_label_set_markup(GTK_LABEL(label),argarg);
            gtk_label_set_line_wrap(GTK_LABEL(label),true);
            gtk_widget_show(label);
            gtk_table_attach(GTK_TABLE(table),label,0,3,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            gtk_misc_set_alignment(GTK_MISC(label),0,0.5);
            found_valid_item = true;
          }

          // Check for a link -> Create GtkLinkButton.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"link")) {
            char label[1024] = { 0 }, url[1024] = { 0 };
            if (std::sscanf(argarg,"%1023[^,],%1023s",label,url)==1) std::strcpy(url,label);
            cimg::strclean(label);
            cimg::strescape(label);
            cimg::strclean(url);
            GtkWidget *link = gtk_link_button_new_with_label(url,label);
            gtk_widget_show(link);
            gtk_table_attach(GTK_TABLE(table),link,0,3,current_line,current_line+1,GTK_FILL,GTK_SHRINK,0,0);
            found_valid_item = true;
          }

          // Check for a value -> No widget but sets parameter value.
          if (!found_valid_item && !cimg::strcasecmp(argtype,"value")) {
            cimg::strclean(argarg);
            set_filter_parameter(filter,current_parameter,argarg);
            found_valid_item = true;
            ++current_parameter;
          }

          if (!found_valid_item) {
            if (get_verbosity_mode()>0)
              std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Found invalid parameter type '%s' for argument '%s'.\n",argtype,argname);
          } else ++current_line;
        } else break;
      }
      set_filter_nbparams(filter,current_parameter);
    }
  }
  gtk_container_add(GTK_CONTAINER(frame),table);
}

// Create main plug-in dialog window and wait for events.
//-------------------------------------------------------
bool create_dialog_gui() {

  // Init GUI_specific variables
  gimp_ui_init("gmic",true);
  event_infos = 0;

  // Create main dialog window with buttons.
  char dialog_title[1024] = { 0 };
  std::sprintf(dialog_title,"%s - %d.%d.%d.%d",
               translate("G'MIC for GIMP"),gmic_version/1000,(gmic_version/100)%10,(gmic_version/10)%10,gmic_version%10);

  GtkWidget *dialog = gimp_dialog_new(dialog_title,"gmic",0,(GtkDialogFlags)0,gimp_standard_help_func,"gmic",NULL);
  gimp_window_set_transient(GTK_WINDOW(dialog));
  gimp_set_data("gmic_gui_dialog",&dialog,sizeof(GtkWidget*));
  g_signal_connect(dialog,"close",G_CALLBACK(on_dialog_cancel_clicked),0);
  g_signal_connect(dialog,"delete-event",G_CALLBACK(on_dialog_cancel_clicked),0);

  GtkWidget *cancel_button = gimp_dialog_add_button(GIMP_DIALOG(dialog),GTK_STOCK_CANCEL,GTK_RESPONSE_CANCEL);
  g_signal_connect(cancel_button,"clicked",G_CALLBACK(on_dialog_cancel_clicked),0);

  GtkWidget *reset_button = gimp_dialog_add_button(GIMP_DIALOG(dialog),GIMP_STOCK_RESET,1);
  g_signal_connect(reset_button,"clicked",G_CALLBACK(on_dialog_reset_clicked),0);
  g_signal_connect_swapped(reset_button,"clicked",G_CALLBACK(on_preview_invalidated),0);

  GtkWidget *apply_button = gimp_dialog_add_button(GIMP_DIALOG(dialog),GTK_STOCK_APPLY,GTK_RESPONSE_APPLY);
  g_signal_connect(apply_button,"clicked",G_CALLBACK(on_dialog_apply_clicked),0);
  g_signal_connect_swapped(apply_button,"clicked",G_CALLBACK(on_preview_invalidated),0);

  GtkWidget *ok_button = gimp_dialog_add_button(GIMP_DIALOG(dialog),GTK_STOCK_OK,GTK_RESPONSE_OK);
  g_signal_connect(ok_button,"clicked",G_CALLBACK(on_dialog_ok_clicked),0);

  GtkWidget *dialog_hbox = gtk_hbox_new(false,0);
  gtk_widget_show(dialog_hbox);
  gtk_container_add(GTK_CONTAINER(GTK_DIALOG(dialog)->vbox),dialog_hbox);

  // Create the left pane.
  left_pane = gtk_vbox_new(false,4);
  gtk_widget_show(left_pane);
  gtk_box_pack_start(GTK_BOX(dialog_hbox),left_pane,true,true,0);

  GtkWidget *image_align = gtk_alignment_new(0.1,0,0,0);
  gtk_widget_show(image_align);
  gtk_box_pack_end(GTK_BOX(left_pane),image_align,false,false,0);
  const unsigned int logo_width = 102, logo_height = 22;
  GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(data_gmic_logo,GDK_COLORSPACE_RGB,false,8,logo_width,logo_height,3*logo_width,0,0);
  GtkWidget *image = gtk_image_new_from_pixbuf(pixbuf);
  gtk_widget_show(image);
  gtk_container_add(GTK_CONTAINER(image_align),image);

  GtkWidget *left_align = gtk_alignment_new(0,0,0,0);
  gtk_widget_show(left_align);
  gtk_box_pack_end(GTK_BOX(left_pane),left_align,false,false,0);

  GtkWidget *left_frame = gtk_frame_new(NULL);
  gtk_widget_show(left_frame);
  gtk_container_set_border_width(GTK_CONTAINER(left_frame),4);
  gtk_container_add(GTK_CONTAINER(left_align),left_frame);

  GtkWidget *frame_title = gtk_label_new(NULL);
  gtk_widget_show(frame_title);
  gtk_label_set_markup(GTK_LABEL(frame_title),translate("<b> Input / Output : </b>"));
  gtk_frame_set_label_widget(GTK_FRAME(left_frame),frame_title);

  GtkWidget *left_table = gtk_table_new(4,1,false);
  gtk_widget_show(left_table);
  gtk_table_set_row_spacings(GTK_TABLE(left_table),6);
  gtk_table_set_col_spacings(GTK_TABLE(left_table),6);
  gtk_container_set_border_width(GTK_CONTAINER(left_table),8);
  gtk_container_add(GTK_CONTAINER(left_frame),left_table);

  input_combobox = gtk_combo_box_new_text();
  gtk_widget_show(input_combobox);
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("Input layers..."));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),"-");
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("None"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("Active (default)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("Active & below"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("Active & above"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All visibles"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All invisibles"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All visibles (decr.)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All invisibles (decr.)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(input_combobox),translate("All (decr.)"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(input_combobox),get_input_mode(false));

  gtk_table_attach_defaults(GTK_TABLE(left_table),input_combobox,0,1,0,1);
  g_signal_connect(input_combobox,"changed",G_CALLBACK(on_dialog_input_mode_changed),0);
  g_signal_connect_swapped(input_combobox,"changed",G_CALLBACK(on_preview_invalidated),0);

  output_combobox = gtk_combo_box_new_text();
  gtk_widget_show(output_combobox);
  gtk_combo_box_append_text(GTK_COMBO_BOX(output_combobox),translate("Output mode..."));
  gtk_combo_box_append_text(GTK_COMBO_BOX(output_combobox),"-");
  gtk_combo_box_append_text(GTK_COMBO_BOX(output_combobox),translate("In place (default)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(output_combobox),translate("New layer(s)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(output_combobox),translate("New image"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(output_combobox),get_output_mode(false));
  gtk_table_attach_defaults(GTK_TABLE(left_table),output_combobox,0,1,1,2);
  g_signal_connect(output_combobox,"changed",G_CALLBACK(on_dialog_output_mode_changed),0);

  preview_combobox = gtk_combo_box_new_text();
  gtk_widget_show(preview_combobox);
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("Output preview..."));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),"-");
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("1st output (default)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("2nd output"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("3rd output"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("4th output"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("1st -> 2nd"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("1st -> 3rd"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("1st -> 4th"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(preview_combobox),translate("All outputs"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(preview_combobox),get_preview_mode(false));
  gtk_table_attach_defaults(GTK_TABLE(left_table),preview_combobox,0,1,2,3);
  g_signal_connect(preview_combobox,"changed",G_CALLBACK(on_dialog_preview_mode_changed),0);
  g_signal_connect_swapped(preview_combobox,"changed",G_CALLBACK(on_preview_invalidated),0);

  verbosity_combobox = gtk_combo_box_new_text();
  gtk_widget_show(verbosity_combobox);
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),translate("Output messages..."));
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),"-");
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),translate("Quiet (default)"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),translate("Verbose"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),translate("Very verbose"));
  gtk_combo_box_append_text(GTK_COMBO_BOX(verbosity_combobox),translate("Debug mode"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(verbosity_combobox),get_verbosity_mode(false));
  gtk_table_attach_defaults(GTK_TABLE(left_table),verbosity_combobox,0,1,3,4);
  g_signal_connect(verbosity_combobox,"changed",G_CALLBACK(on_dialog_verbosity_mode_changed),0);
  g_signal_connect_swapped(verbosity_combobox,"changed",G_CALLBACK(on_preview_invalidated),0);

  drawable_preview = gimp_drawable_get(gimp_image_get_active_drawable(image_id));
  gui_preview = gimp_zoom_preview_new(drawable_preview);
  gtk_widget_show(gui_preview);
  gtk_box_pack_end(GTK_BOX(left_pane),gui_preview,true,true,0);
  g_signal_connect(gui_preview,"invalidated",G_CALLBACK(process_preview),0);

  // Create the middle pane.
  GtkWidget *middle_frame = gtk_frame_new(NULL);
  gtk_widget_show(middle_frame);
  gtk_container_set_border_width(GTK_CONTAINER(middle_frame),4);
  gtk_box_pack_start(GTK_BOX(dialog_hbox),middle_frame,false,false,0);

  GtkWidget *middle_pane = gtk_vbox_new(false,4);
  gtk_widget_show(middle_pane);
  gtk_container_add(GTK_CONTAINER(middle_frame),middle_pane);

  GtkWidget *scrolledwindow = gtk_scrolled_window_new(NULL,NULL);
  gtk_widget_show(scrolledwindow);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolledwindow),GTK_POLICY_AUTOMATIC,GTK_POLICY_AUTOMATIC);
  gtk_box_pack_start(GTK_BOX(middle_pane),scrolledwindow,true,true,0);

  GtkWidget *treeview = gtk_tree_view_new_with_model(GTK_TREE_MODEL(treeview_store));
  gtk_widget_show(treeview);
  gtk_container_add(GTK_CONTAINER(scrolledwindow),treeview);

  GtkWidget *tree_hbox = gtk_hbox_new(false,6);
  gtk_widget_show(tree_hbox);
  gtk_box_pack_start(GTK_BOX(middle_pane),tree_hbox,false,false,0);

  GtkWidget *update_button = gtk_button_new_from_stock(GTK_STOCK_REFRESH);
  gtk_widget_show(update_button);
  gtk_box_pack_start(GTK_BOX(tree_hbox),update_button,false,false,0);
  g_signal_connect(update_button,"clicked",G_CALLBACK(on_dialog_update_clicked),(void*)treeview);

  GtkWidget *internet_checkbutton = gtk_check_button_new_with_mnemonic(translate("_Internet updates"));
  gtk_widget_show(internet_checkbutton);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(internet_checkbutton),get_net_update());
  gtk_box_pack_start(GTK_BOX(tree_hbox),internet_checkbutton,false,false,0);
  g_signal_connect(internet_checkbutton,"toggled",G_CALLBACK(on_dialog_net_update_toggled),(void*)internet_checkbutton);

  treemode_button = gtk_button_new();
  gtk_box_pack_start(GTK_BOX(tree_hbox),treemode_button,false,false,0);
  g_signal_connect(treemode_button,"clicked",G_CALLBACK(on_dialog_treemode_clicked),(void*)treeview);

  GtkTreeViewColumn *column = gtk_tree_view_column_new();
  gtk_tree_view_append_column(GTK_TREE_VIEW(treeview),column);
  flush_treeview(treeview);

  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));
  gtk_tree_selection_set_mode(selection,GTK_SELECTION_SINGLE);
  g_signal_connect(G_OBJECT(selection),"changed",G_CALLBACK(on_filter_changed),0);
  g_signal_connect_swapped(selection,"changed",G_CALLBACK(on_preview_invalidated),0);

  // Create the right pane.
  GtkWidget *right_pane = gtk_vbox_new(false,0);
  gtk_widget_show(right_pane);
  gtk_box_pack_start(GTK_BOX(dialog_hbox),right_pane,false,false,0);

  GtkWidget *right_frame = gtk_frame_new(NULL);
  gtk_widget_show(right_frame);
  gtk_container_set_border_width(GTK_CONTAINER(right_frame),4);
  gtk_widget_set_size_request(right_frame,450,-1);
  gtk_box_pack_start(GTK_BOX(right_pane),right_frame,true,true,0);
  gimp_set_data("gmic_gui_frame",&right_frame,sizeof(GtkWidget*));
  create_parameters_gui(false);

  // Show dialog window and wait for user response.
  gtk_widget_show(dialog);
  gtk_main();

  // Destroy dialog box widget and free resources.
  gtk_widget_destroy(dialog);
  if (treemode_stockbutton) gtk_widget_destroy(treemode_stockbutton);
  if (event_infos) delete[] event_infos;
  return return_create_dialog;
}

// 'Run' function, required by the GIMP plug-in API.
//--------------------------------------------------
void gmic_run(const gchar *name, gint nparams, const GimpParam *param, gint *nreturn_vals, GimpParam **return_vals) {
  bool is_existing_instance = false;

  // Init plug-in variables.
  try {
    set_locale();
    static GimpParam values[1];
    values[0].type = GIMP_PDB_STATUS;
    *return_vals  = values;
    *nreturn_vals = 1;
    name = 0;
    nparams = 0;
    GimpRunMode run_mode;
    run_mode = (GimpRunMode)param[0].data.d_int32;
    if (run_mode==GIMP_RUN_NONINTERACTIVE) {
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : ERROR, this plug-in cannot be run in non-interactive mode.\n");
      values[0].data.d_status = GIMP_PDB_CALLING_ERROR;
      return;
    }
    return_create_dialog = true;
    gimp_get_data("gmic_instance",&is_existing_instance);
    if (is_existing_instance) {
      std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Existing instance of the plug-in seems to be already running.\n");
      return;
    }
    is_existing_instance = true;
    gimp_set_data("gmic_instance",&is_existing_instance,sizeof(bool));
    update_filters_definition(false);

    // Get active image.
    image_id = gimp_drawable_get_image(param[2].data.d_drawable);
    gimp_tile_cache_ntiles(2*(gimp_image_width(image_id)/gimp_tile_width()+1));

    if (run_mode==GIMP_RUN_INTERACTIVE) { // If interactive mode, show the dialog window.
      if (create_dialog_gui()) {
        process_image(0);
        const char *commandline = get_commandline(false);
        if (commandline) { // Remember command line for the next use of the filter.
          char s_tmp[256] = { 0 };
          std::sprintf(s_tmp,"gmic_commandline%u",get_current_filter());
          gimp_set_data(s_tmp,commandline,std::strlen(commandline));
        }
      }
    } else if (run_mode==GIMP_RUN_WITH_LAST_VALS) { // If non interactive mode, run the last used filter.
      const unsigned int filter = get_current_filter();
      if (filter) {
        char s_tmp[256] = { 0 };
        std::sprintf(s_tmp,"gmic_commandline%u",filter);
        char commandline[4096] = { 0 };
        gimp_get_data(s_tmp,&commandline);
        process_image(commandline);
      }
    }

    // Free plug-in resources.
    delete[] gmic_macros;
    values[0].data.d_status = GIMP_PDB_SUCCESS;
    is_existing_instance = false;
    gimp_set_data("gmic_instance",&is_existing_instance,sizeof(bool));
  } catch (CImgException &e) {
    std::fprintf(stderr,"\n*** Plug-in 'gmic4gimp' : Fatal error encountered in plug-in code :\n*** %s\n",e.message);
    is_existing_instance = false;
    gimp_set_data("gmic_instance",&is_existing_instance,sizeof(bool));
  }
}

// 'Query' function, required by the GIMP plug-in API.
//----------------------------------------------------
void gmic_query() {
  static const GimpParamDef args[] = {
    {GIMP_PDB_INT32,    (gchar*)"run_mode", (gchar*)"Run mode"},
    {GIMP_PDB_IMAGE,    (gchar*)"image",    (gchar*)"Input image"},
    {GIMP_PDB_DRAWABLE, (gchar*)"drawable", (gchar*)"Input drawable"}
  };

  set_locale();
  gimp_install_procedure("gmic",                     // name
                         "G'MIC",                    // blurb
                         "G'MIC",                    // help
                         "David Tschumperle",        // author
                         "David Tschumperle",        // copyright
                         "2008-12-02",               // date
                         translate("_G'MIC for GIMP..."), // menu_path
                         "RGB*, GRAY*",              // image_types
                         GIMP_PLUGIN,                // type
                         G_N_ELEMENTS(args),         // nparams
                         0,                          // nreturn_vals
                         args,                       // params
                         0);                         // return_vals

  gimp_plugin_menu_register("gmic", "<Image>/Filters");
}

GimpPlugInInfo PLUG_IN_INFO = { 0, 0, gmic_query, gmic_run };
MAIN();
