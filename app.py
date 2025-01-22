import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load data
students_df = pd.read_csv("students.csv")
clubs_df = pd.read_csv("clubs.csv")

# Preprocess student interests
students_df['CombinedInterests'] = students_df[['Interest1', 'Interest2', 'Interest3']].apply(
    lambda row: ' '.join(row.values.astype(str)), axis=1
)

# Vectorize interests
vectorizer = CountVectorizer()
interest_vectors = vectorizer.fit_transform(students_df['CombinedInterests'])

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
students_df['Cluster'] = kmeans.fit_predict(interest_vectors)

# Recommendation function
def recommend_clubs(student_ids):
    selected_students = students_df[students_df['StudentID'].isin(student_ids)]
    student_clusters = selected_students['Cluster'].unique()
    similar_students = students_df[students_df['Cluster'].isin(student_clusters)]
    
    # Aggregate interests of similar students
    all_interests = ' '.join(similar_students['CombinedInterests'])
    common_interests = pd.Series(all_interests.split()).value_counts()
    
    # Recommend clubs based on common interests
    recommended_clubs = clubs_df[clubs_df['RelatedInterest'].isin(common_interests.index)]
    recommended_clubs['Frequency'] = recommended_clubs['RelatedInterest'].map(common_interests)
    return recommended_clubs.sort_values(by='Frequency', ascending=False)

def get_all_club_clusters():
    # Assign a cluster to each club based on its related interest and the popularity of the interest
    club_interest_mapping = clubs_df.groupby('RelatedInterest').size().reset_index(name='Count')
    club_interest_mapping['Cluster'] = club_interest_mapping.groupby('Count').ngroup()  # Assign clusters
    return club_interest_mapping

def get_club_leaderboard():
    # Calculate the demand for each club based on how many students have similar interests
    club_demand = clubs_df.groupby('RelatedInterest').size().reset_index(name='Demand')
    club_demand = club_demand.sort_values(by='Demand', ascending=False).reset_index(drop=True)
    return club_demand

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    # Navbar
    html.Nav([
        html.A("Home", href="#", style={'color': 'white', 'margin': '0 15px', 'font-weight': 'bold'}),
        html.A("Dashboard", href="#dashboard", style={'color': 'white', 'margin': '0 15px', 'font-weight': 'bold'}),
        html.A("About Us", href="#about-us", style={'color': 'white', 'margin': '0 15px', 'font-weight': 'bold'}),
    ], style={'background-color': '#1a1a1a', 'padding': '10px', 'display': 'flex', 'justify-content': 'center'}),

    # Landing Page Section
    html.Section([
        html.Div([
            html.Div([
                html.H1("Welcome to College Club Engagement Tracker", style={'font-size': '36px', 'color': '#fff', 'text-align': 'center'}),
                html.P("Find the best clubs based on your interests and connect with like-minded students!", 
                       style={'font-size': '18px', 'color': '#ccc', 'text-align': 'center'}),

                # Add fewer lines above the Get Started button
                html.P("Discover clubs that resonate with your interests.", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
                html.P("Join and connect with students who share your passions.", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),

                # Add image centered above the "Get Started" button
                html.Div([
                    html.Img(src="https://img.freepik.com/free-vector/student-club-abstract-concept-vector-illustration-student-organization-university-interest-club-afterschool-activity-program-college-association-professional-hobby-society-abstract-metaphor_335657-5900.jpg", alt="Placeholder Image", style={'width': '360px', 'height': 'auto', 'display': 'block', 'margin': '0 auto'}),
                ], style={'text-align': 'center', 'margin-bottom': '20px'}),

                html.Button("Get Started", id="get-started", style={
                    'background-color': '#007BFF', 'color': 'white', 'padding': '15px 30px', 'border': 'none',
                    'border-radius': '5px', 'font-size': '18px', 'cursor': 'pointer', 'transition': 'background-color 0.3s ease'}),
            ], style={'flex': 1, 'padding': '20px', 'text-align': 'center'}),
        ], style={'background-color': '#222', 'display': 'flex', 'flex-direction': 'column', 'padding': '40px 20px',
                  'background-image': 'url("https://img.freepik.com/premium-vector/vector-dynamic-line-dark-technology-background_93566-14.jpg")', 'background-size': 'cover', 'background-position': 'center'}),
    ]),

    # Dashboard Section
    html.Section([
        html.Div([
            html.H2("Dashboard", id="dashboard", style={'font-size': '30px', 'color': '#fff', 'margin-bottom': '30px', 'text-align': 'center'}),
            html.Label("Select Students:", style={'color': '#fff', 'font-size': '18px'}),
            dcc.Dropdown(
                id='student-dropdown',
                options=[
                    {'label': f"{row['Name']} (ID: {row['StudentID']})", 'value': row['StudentID']}
                    for _, row in students_df.iterrows()
                ],
                multi=True,
                placeholder="Select one or more students",
                style={'width': '80%', 'margin': '0 auto', 'display': 'block', 'padding': '10px', 'font-size': '16px'}
            ),
        ], style={'text-align': 'center', 'padding': '20px'}),

        html.Div(id='recommendation-output', style={'margin-top': '20px'}),
        
        html.Div([
            html.H3("Interest Distribution in Selected Cluster", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
            dcc.Graph(id='interest-pie-chart', style={'margin-bottom': '30px'}),
            
            html.H3("Club Popularity", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
            dcc.Graph(id='club-bar-chart', style={'margin-bottom': '30px'}),
            
            html.H3("All Club Clusters", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
            dcc.Graph(id='all-club-clusters', style={'margin-bottom': '30px'}),

            # Leaderboard for Most Demanding Clubs
            html.H3("Most Demanding Clubs - Leaderboard", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
            dcc.Graph(id='club-leaderboard', style={'margin-bottom': '30px'}),
            
            # Section for Club Selection and Students Data
            html.H3("Select a Club to View Students Interested in it", style={'font-size': '20px', 'color': '#fff', 'text-align': 'center'}),
            dcc.Dropdown(
                id='club-dropdown',
                options=[
                    {'label': club, 'value': club}
                    for club in clubs_df['ClubName'].unique()
                ],
                placeholder="Select a club",
                style={'width': '80%', 'margin': '0 auto', 'display': 'block', 'padding': '10px', 'font-size': '16px'}
            ),
            
            # Table for displaying student data based on selected club
            html.Div(id='student-table', style={'margin-top': '30px'})
        ], style={'text-align': 'center'}),
    ], style={'background-color': '#222'}),

    # Footer
    html.Footer([
        html.P("© 2025 College Club Engagement Tracker | All Rights Reserved.", style={'text-align': 'center', 'color': '#fff', 'padding': '20px', 'background-color': '#1a1a1a'}),
    ], style={ 'width': '100%', 'bottom': '0'})
])

# Callbacks
@app.callback(
    Output('recommendation-output', 'children'),
    Input('student-dropdown', 'value')
)
def update_recommendations(selected_students):
    if not selected_students:
        return html.Div("Select one or more students to see recommendations.", style={'color': '#d9534f'})
    
    recommendations = recommend_clubs(selected_students)
    return html.Div([
        html.H3("Club Recommendations Based on Your Interests", style={'color': '#fff'}),
        html.Table([
            html.Tr([html.Th("Club Name"), html.Th("Related Interest"), html.Th("Popularity")])
        ] + [
            html.Tr([html.Td(row['ClubName']), html.Td(row['RelatedInterest']), html.Td(row['Frequency'])])
            for _, row in recommendations.iterrows()
        ], style={'margin': '0 auto', 'border-collapse': 'collapse'})
    ], style={'color': '#fff'})

@app.callback(
    Output('interest-pie-chart', 'figure'),
    Input('student-dropdown', 'value')
)
def update_interest_pie(selected_students):
    if not selected_students:
        return go.Figure()

    selected_students_data = students_df[students_df['StudentID'].isin(selected_students)]
    interest_counts = selected_students_data['CombinedInterests'].str.split().explode().value_counts()

    return px.pie(values=interest_counts.values, names=interest_counts.index, title="Interest Distribution")

@app.callback(
    Output('club-bar-chart', 'figure'),
    Input('student-dropdown', 'value')
)
def update_club_bar_chart(selected_students):
    if not selected_students:
        return go.Figure()

    recommendations = recommend_clubs(selected_students)
    return px.bar(recommendations, x='ClubName', y='Frequency', title="Club Popularity")

@app.callback(
    Output('all-club-clusters', 'figure'),
    Input('student-dropdown', 'value')
)
def update_all_club_clusters(selected_students):
    club_clusters = get_all_club_clusters()
    return px.scatter(club_clusters, x='Count', y='Cluster', color='Cluster', title="All Club Clusters")

@app.callback(
    Output('club-leaderboard', 'figure'),
    Input('student-dropdown', 'value')
)
def update_club_leaderboard(selected_students):
    club_leaderboard = get_club_leaderboard()
    return px.bar(club_leaderboard, x='RelatedInterest', y='Demand', title="Most Demanding Clubs")

@app.callback(
    Output('student-table', 'children'),
    Input('club-dropdown', 'value')
)
def update_student_table(selected_club):
    if not selected_club:
        return html.Div("Select a club to view students interested in it.", style={'color': '#d9534f'})
    
    students_in_club = students_df[students_df['CombinedInterests'].str.contains(selected_club)]
    return html.Table([
        html.Tr([html.Th("Student Name"), html.Th("Interest")])
    ] + [
        html.Tr([html.Td(row['Name']), html.Td(row['CombinedInterests'])])
        for _, row in students_in_club.iterrows()
    ], style={'margin': '0 auto', 'border-collapse': 'collapse'})

if __name__ == '__main__':
    app.run_server(debug=True)
