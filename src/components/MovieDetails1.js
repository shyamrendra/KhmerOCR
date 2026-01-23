import React, {Component} from 'react'
import axios from 'axios'
import {Route} from 'react-router-dom'



class MovieDetails extends Component {

      constructor(props){
         super(props) ;

         this.state ={

             photoDetails:[]

         } ; 

      }


   


   componentDidMount(){

   	   const { match } = this.props;
       axios.get(`https://jsonplaceholder.typicode.com/photos/${match.params.photo_id}`)

       .then(response => {

       	this.setState({

       		photoDetails:response.data
       	})


       })


       	

   }


      render(){

      	

         return(

           


              
         	)

      }

} 

export default MovieDetails;