import React, {Component} from 'react'
import axios from 'axios'
import {Route, NavLink} from 'react-router-dom'



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

      	 const { photoDetails } = this.state;
         
         return(

             
            <div>
             
             <NavLink to="/">Back</NavLink>


            


           <div key ={photoDetails.id}className = "col-md-3">               

                <div className="card">
                 
                 <img src ={photoDetails.thumbnailUrl} className ="card-img-top" alt ="" />

                <div className="card-body">
                <h4 className="card-title">{photoDetails.title}</h4>
                 
                
                 </div>
                 
                 

                 </div>



               </div>


               </div>

          )

            
              
         	

      }

} 

export default MovieDetails;