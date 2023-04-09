from flask import Flask, request, render_template
import recommendations

app = Flask(_name_)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    job_description = request.form['job_description']
    user_resume = request.form['user_resume']
    recommended_jobs = recommendations.generate_recommendations(job_description, user_resume)
    return render_template('recommendations.html', jobs=recommended_jobs)