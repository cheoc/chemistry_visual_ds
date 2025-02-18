import pathlib
import dash
import sys 
import logging
import io
import csv

from io import StringIO
from dash import dcc, html, no_update, dash_table
from tqdm import tqdm

import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
from pandas.api.types import is_numeric_dtype


import base64
import pandas as pd
import plotly.graph_objs as go

from pandas import ExcelFile

from helpers import helper_general
from helpers import helper_chemistry

# sys.path.insert(1, '/home/ch2j/code/helpers')
# import helper_molecules
#import helper_classifier

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

my_border_style = ""

app.layout = dbc.Container(fluid=True, children=[
    html.H1("Dataset annotation", id="nav-pills"),
    dcc.Store(id='memory-output'),
    dcc.Store(id='memory-output-filtered'),
    dcc.Store(id='memory-output-pains'),
    dcc.Store(id='memory-output-magic-rings'),    
    dcc.Store(id='memory-output-matched-pains'),
    dcc.Store(id='memory-output-matched-magic-rings'),     
    dcc.Store(id='memory-save-molecules'),
    html.Br(),
    html.Div([
        html.H6("Upload molecules (needs to contain a smiles columns)"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                #"margin": "10px",
            },
            multiple=False,
        ),
        
        dcc.Loading(
            children=[html.Div(id='output-loaded-file', style={'width':'auto'})],
        ),
        html.Br(),
        # dbc.Row([
        #     dcc.Dropdown(['PAINS', 'Magic rings', 'Lipinski Ro5'], 'PAINS', id='filters-dropdown')
        # ]),
        html.Button("Clear selection", id="table-clear"),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(          
                            children = ["Overview", 
                                        html.Div(children=[dash_table.DataTable(id='table-main')],
                                                 style={'width':'auto'})
                                        ]
                        )
                    ], style={"height": "100%", "borderStyle": my_border_style}),                                                
                    dbc.Col([
                        dcc.Loading(          
                            children = ["PAINS", 
                                        html.Div(children=[dash_table.DataTable(id='table-pains')],
                                                 style={'width':'auto'})
                                       ]
                        )
                    ], style={"height": "100%", "borderStyle": my_border_style}),                                                         
                    dbc.Col([
                        dcc.Loading(          
                            children = ["Magic rings", 
                                        html.Div(children=[dash_table.DataTable(id='table-magic')], 
                                                 style={'width':'auto'})
                            ]
                        )
                    ], style={"height": "100%", "borderStyle": my_border_style}),                             
                ]),                      
                dbc.Row([dbc.Col(children=["Molecules to export", 
                                            html.Div(children=[
                                                dash_table.DataTable(id='table-molecules',
                                                                     css=[dict(selector="p", rule="margin: 0px; text-align: center")],
                                                                     style_cell={"textAlign": "center"},
                                                                     markdown_options={"html": True},
                                                                     sort_action="native",
                                                                     page_size=10
                                                                    ),
                                                                     
                                            ]),
                                            html.Button('Download data', id='download-data', n_clicks=0),
                                            dcc.Download(id="download-dataframe-xlsx"),
                                            ],
                            style={"height": "100%", "borderStyle": my_border_style}) 
                ]),
            ]),                        
        ]),       
    ])
])

@app.callback(
    [Output("table-main", "selected_cells"),
     Output("table-main", "active_cell"),
     Output("table-pains", "selected_cells"),
     Output("table-pains", "active_cell"),
     Output("table-magic", "selected_cells"),
     Output("table-magic", "active_cell"),
    ],
    Input("table-clear", "n_clicks"),    
)
def table_clear(n_clicks):
    return [], None, [], None, [], None

@app.callback([Output('memory-output', 'data'), 
               Output('memory-output-pains', 'data'),
               Output('memory-output-magic-rings', 'data'),
               Output('output-loaded-file', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    
    df = None
    
    if contents is not None:

        try:
            content_type, content_string = contents.split(',')
            
            file_extension = pathlib.Path(filename).suffix.lower()

            df = pd.DataFrame()
            if file_extension == ".txt" or file_extension == ".csv":                        
                decoded = base64.b64decode(content_string).decode() 
                
                #sniffer = csv.Sniffer()   
                # delimiter = sniffer.sniff(decoded).delimiter    
                # print(delimiter.string)          
                try:            
                    df = pd.read_csv(StringIO(decoded), sep=",|;", engine="python") 
                except:
                    try:
                        df = pd.read_csv(StringIO(decoded), sep=None, engine='python') 
                    except:
                        return "smiles column not found", filename
                          
            elif file_extension == ".xlsx":
                decoded = base64.b64decode(content_string)
                df = pd.read_excel(io.BytesIO(decoded))
            
            col_smiles = ["smiles", "SMILES", "mol", "MOL"]
            smiles_default_name = "smiles"
            
            for col_name in col_smiles:
                if col_name in df:
                    df[smiles_default_name] = df[col_name]
            
            if smiles_default_name not in df:
                return None, None, None, "smiles column not found"
                                                
            tl = helper_chemistry.get_list_physchem_props_from_smiles(df[smiles_default_name])


            for i in tqdm(range(len(tl))):
                props = tl[i]
                for t, k in props.items():
                    if t not in df:
                        df[t] = None
                    df.loc[i, t] = k

        except Exception as ex:
            print("ERROR reading file", str(ex))
            #table = html.Div(['There was an error processing this file.'])
        
        if filename is None:
            filename = "No file loaded"
        
        #### read pains
        
        try:
            print("Loading pains")
            
            df_pains = pd.read_csv("../data/pains_smarts.txt", sep="\t")
             
        except Exception as ex:
            print(str(ex))
        
        #### read magic rings
        
        try:
            print("Loading magic rings")
            
            df_magic = pd.read_csv("../data/magic_rings.txt", sep="\t")
              
        except Exception as ex:
            print(str(ex))        
        
        logging.warning("molecules loaded")
        
        
        #
        
        return df.to_dict('records'), df_pains.to_dict('records'), df_magic.to_dict('records'), filename
    
    return None, None, None, "no file loaded"

@app.callback(Output('table-main', 'data'),
              Input('memory-output', 'data'))
def on_data_set_table(data):
    
    if data is None:
        return None
    else:
        try:

            df = pd.DataFrame(data)
                                
            tl = [
                {"property": "#molecules", "value": len(df)},
                {"property": "#scaffolds (murcko)", "value": len(df["scaffold_smiles"].unique())},
                {"property": "#scaffolds (generic)", "value": len(df["scaffold_generic_smiles"].unique())},
                {"property": "#Lipinski Ro5", "value": len(helper_chemistry.get_number_lipinski_ro5(df))},
                {"property": "#druglikeness (Ghose)", "value": len(helper_chemistry.get_number_druglikeness_ghose(df))},
                {"property": "#druglikeness (Veber)", "value": len(helper_chemistry.get_number_druglikeness_veber(df))}
            ]                
            
            df_show = pd.DataFrame(tl)                       
            
            #table = dash_table.DataTable()
            return df_show.to_dict('records')
            
        except Exception as e:
            print("Error processing data set", str(e))
        
        return None

@app.callback([Output('table-pains', 'data'),
               Output('memory-output-matched-pains', 'data')],
              Input('memory-output', 'data'),
              State('memory-output-pains', 'data'))
def on_data_set_table_pains(data, data_pains):
    
    if data is None or data_pains is None:
        return None, []
    else:
        try:

            df = pd.DataFrame(data)
            df_pains = pd.DataFrame(data_pains)     

            tl_matches = helper_chemistry.annotate_pains(df, df_pains)
            
            #print(df[(df["pains"] != "")]["pains"])
            
            nof_pains = len(df[df["pains"] != ""]) 
            nof_no_pains = len(df[df["pains"] == ""])
            nof_unique_pains = len(df[(df["pains"] != "")]["pains"].unique())
            
            tl = [
                {"property": "#pains", "value": len(df_pains)},
                {"property": "#matches pains", "value": nof_pains},                
                {"property": "#matches no pains", "value": nof_no_pains},                
                {"property": "#unique matched pains", "value": nof_unique_pains},
            ]                
                                    
            df_show = pd.DataFrame(tl)                       
            
            return df_show.to_dict('records'), tl_matches
            
        except Exception as e:
            print("Error processing data set", str(e))
        
        return None, []

@app.callback([Output('table-magic', 'data'),
               Output('memory-output-matched-magic-rings', 'data')],
              Input('memory-output', 'data'),
              State('memory-output-magic-rings', 'data'))
def on_data_set_table_magic(data, data_magic):
    
    if data is None or data_magic is None:
        return None, []
    else:
        try:

            df = pd.DataFrame(data)
            df_magic = pd.DataFrame(data_magic)     

            tl_matches = helper_chemistry.annotate_magic_rings(df, df_magic)                        
            
            tl_match_ids = []
            count = 0
            dict_class_count = {}
            dict_activity_count = {}
            for matches in tl_matches:
                for match in matches:
                    tl_match_ids.append(match[1])
                    count += 1
                    cur_class = df_magic["class"].iloc[match[1]]
                    if cur_class not in dict_class_count:
                        dict_class_count[cur_class] = 0
                    dict_class_count[cur_class] += 1
                    
                    cur_act = df_magic["activity"].iloc[match[1]]
                    if cur_act != '-':
                        if cur_act not in dict_activity_count:
                            dict_activity_count[cur_act] = 0
                        dict_activity_count[cur_act] += 1                    
                                                        
            nof_matched_classes = len(set(tl_match_ids))

            tl = [
                {"property": "#magic rings", "value": len(df_magic)},
                {"property": "#molecules with hits", "value": len(tl_matches)},
                {"property": "#rings with hits", "value": nof_matched_classes},
                {"property": "#total matches", "value": count}, 
            ]                
            for t,k in dict_class_count.items():
                tl.append({"property": "#matched classes: "+t, "value": k})
                
            for t,k in dict_activity_count.items():
                tl.append({"property": "#matched activity: "+t, "value": k})            
                       
            df_show = pd.DataFrame(tl)                       
            
            #table = dash_table.DataTable(df_show.to_dict('records'))
            return df_show.to_dict('records'), tl_matches
            
        except Exception as e:
            print("Error processing data set", str(e))
        
        return None, []

@app.callback([Output('table-molecules', 'data'), 
               Output('table-molecules', 'columns'), 
               Output('memory-save-molecules', 'data')],
              [Input('table-main', 'active_cell'),
               Input('table-pains', 'active_cell'),
               Input('table-magic', 'active_cell')],
              [State('memory-output', 'data'), 
               State('memory-output-pains', 'data'),
               State('memory-output-magic-rings', 'data'),
               State('memory-output-matched-pains', 'data'),
               State('memory-output-matched-magic-rings', 'data'),
               State('table-magic', 'data')
               ])
def update_molecules_table(t_main_ac, t_pains_ac, t_magic_ac, data, data_pains, data_magic, tl_matched_pains, tl_matched_magic, table_magic):
    
    print(t_main_ac, t_pains_ac, t_magic_ac)
    
    if data is None or\
       data_pains is None or\
       data_magic is None or\
       tl_matched_pains is None or\
       tl_matched_magic is None:
        return None, None, None
    
    try:
        print("set mols")
                
        df = pd.DataFrame(data)
        df_pains = pd.DataFrame(data_pains)
        df_magic = pd.DataFrame(data_magic)
        
           
        #         tl_df = []
        #         for tmp_dict in selectedData["points"]:
        #             tl_df.append(tmp_dict["pointNumber"])
                            
        #         df_new = pd.DataFrame(df.iloc[tl_df])        
                
        # df_new["molecule"] = [helper_chemistry.smi2svg(smi, x=100, y=100) for smi in df_new["smiles"]]
                
        #         tl_selection = ["molecule", "LogP", "Molecular weight"]
    
        df_show = pd.DataFrame()
        
        tl_main_ids = []
        tl_pains_ids = []
        tl_pains_container = []
        tl_magic_ids = []
        dict_magic_container = {}
        if t_main_ac is not None:
           
            if t_main_ac["row"] == 0:
                tl_main_ids = [x for x in range(len(df))]
            elif t_main_ac["row"] == 1:
                pass
            elif t_main_ac["row"] == 2:
                pass
            elif t_main_ac["row"] == 3:   
                tl_main_ids = helper_chemistry.get_number_lipinski_ro5(df)
            elif t_main_ac["row"] == 4:
                tl_main_ids = helper_chemistry.get_number_druglikeness_ghose(df)
            elif t_main_ac["row"] == 5:
                tl_main_ids = helper_chemistry.get_number_druglikeness_veber(df)
        
        if t_pains_ac is not None:
            print(tl_matched_pains)
            #if len(tl_matched_pains) > 0:
            print("pains")
            #print(tl_pains[0])
        
        if t_magic_ac is not None:
            print("magic")
            selection = table_magic[t_magic_ac['row']]['property']
            sel_prop = ""
            magic_prop = ""
            if "matched classes" in selection:
                sel_prop = selection.split(":")[1].strip()
                magic_prop = "class"
            elif "matched activity" in selection:
                sel_prop = selection.split(":")[1].strip()
                magic_prop = "activity"
            
            print(magic_prop)
            print(sel_prop)            
            print(tl_matched_magic)
            
            for tl_mol in tl_matched_magic:
                print("################################")
                for tl_tmp_match in tl_mol:
                    print("start", tl_tmp_match)
                    mol_id = tl_tmp_match[0]
                    scaff_id = tl_tmp_match[1]
                    print(mol_id, scaff_id, len(df_magic))
                    if df_magic[magic_prop].iloc[scaff_id] == sel_prop:
                        print("sel_prop", df_magic[magic_prop].iloc[scaff_id], sel_prop)
                        tl_magic_ids.append(mol_id)
                        if mol_id not in dict_magic_container:
                            dict_magic_container[mol_id] = []
                            
                        dict_magic_container[mol_id].append(tl_tmp_match)
                        print("completed")    
        
        print("testing")

        set_selection_tmp = set()
        if len(tl_main_ids) > 0:
            set_selection_tmp = set(tl_main_ids)
        
        if len(tl_pains_ids) > 0:
            if len(set_selection_tmp) > 0:
                set_selection_tmp = set(set_selection_tmp) & set(tl_pains_ids)
            else:
                set_selection_tmp = set(tl_pains_ids)
        
        if len(tl_magic_ids) > 0:
            if len(set_selection_tmp) > 0:
                set_selection_tmp = set(set_selection_tmp) & set(tl_magic_ids)
            else:
                set_selection_tmp = set(tl_magic_ids)        
                
        
        tl_selection = [x for x in set_selection_tmp]
        

        print(len(tl_selection))

        if len(tl_selection) > 0:
            df_show = df.iloc[tl_selection].copy()
            df_show["ID"] = [i for i in tl_selection]
            df_show["molecule"] = [helper_chemistry.smi2svg(smi, x=100, y=100) for smi in df_show["smiles"]]
            df_show["magic_rings"] = ["" for i in tl_selection]
            
            if len(tl_magic_ids) > 0:
                print("m2s select", tl_selection)
                df_show["magic_rings"] = [helper_chemistry.smi_match2svg(df["smiles"].iloc[i], dict_magic_container[i], x=100, y=100) for i in tl_selection]

                print(df_show["magic_rings"].head(2))
            
            columns=[
                {"name": col, "id": col, "presentation": "markdown"}
                if col == "molecule"
                else {"name": col, "id": col}
                for col in df_show[["ID", "molecule", "Molecular weight"]].columns
            ]            
            
            return df_show.to_dict("records"), columns, None
            
        else:
            return None, None, None
        
    except Exception as ex:
        print(str(ex))
        
    return None, None, None

# @app.callback(Output('output-data-table', 'children'),
#               Input('memory-output-filtered', 'data'))
# def on_data_set_table(data):
    
#     if data is None:
#         return ""
#     else:
#         try:

#             df = pd.DataFrame(data)                                             
            
#             #df_show = pd.DataFrame(tl)                       
            
#             #table = dash_table.DataTable(df.to_dict('records'))
            
#             df["molecule"] = helper_chemistry.generate_molecule_images([x for x in df["smiles"]])
        
#             tl_selection = ["molecule", "pains", "Molecular weight"]
        
#             if len(df) > 0:
#                 table = dash_table.DataTable(
#                     css=[dict(selector="p", rule="margin: 0px; text-align: center")],
#                     data=df[tl_selection].to_dict("records"),
#                     style_cell={"textAlign": "center"},
#                     columns=[
#                         {"name": col, "id": col, "presentation": "markdown"}
#                         if col == "molecule"
#                         else {"name": col, "id": col}
#                         for col in df[tl_selection].columns
#                     ],
#                     markdown_options={"html": True},
#                     sort_action="native",
#                     page_size=10
#                 )        
                
#                 return table            
                        
#             return "no data"
            
#         except Exception as e:
#             print("Error processing data set", str(e))
        
#         return "something went wrong"

@app.callback(
    Output('download-dataframe-xlsx', 'data'),
    Input('download-data', 'n_clicks'),
    State('memory-save-molecules', 'data')
)
def update_output(n_clicks, data):
    
    try:
        if data is not None:    
            
            df = pd.DataFrame(data) 
        
            return dcc.send_data_frame(df.to_excel, "mydf.xlsx", sheet_name="Selected_data")
        
    except Exception as ex:
        print(str(ex))
    
    


if __name__ == '__main__':
    # import socket
    # hostname = socket.gethostname()
    # IPAddr = socket.gethostbyname(hostname)            
    
    app.run_server(debug=True,host='0.0.0.0', port=8144)