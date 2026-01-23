import React, {Component} from 'react'
import {Route, Switch} from 'react-router-dom'

import MovieList from './components/MovieList'
import MovieDetails from './components/MovieDetails'


class App extends Component {




 render() { 
    return(

       <section>
       <Switch>

	     <Route exact path = '/' component = {MovieList} />
	     <Route path = '/:photo_id' component = {MovieDetails} />

	   </Switch>

	   </section>

    );
    
      
  }
  
}

export default App;
