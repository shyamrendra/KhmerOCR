import React, { Component } from 'react';
import axios from 'axios';
import { NavLink } from 'react-router-dom'
import { Card, CardImg, CardText, CardBody,  CardTitle, CardSubtitle, Button } from 'reactstrap'
import './default.css';



class MovieList extends Component {

       constructor(props){
             super(props)

             this.state = {

                photos:[],
                visiblePhotos:[]

             }




       }





   
    componentDidMount(){

    	axios.get("https://jsonplaceholder.typicode.com/photos")
    	.then((response) =>{

    		console.log(response.data);
    		this.setState({

                photos:response.data.slice(0,20),
                visiblePhotos:response.data.slice(0,20)

    		})


    	});


    		
    }


filterPhotos = () => {

  console.log(this.filterText.value);
  const filterPhotoList = this.state.photos.filter((photo) => {

    return photo.title.indexOf(this.filterText.value) !== -1


  });
          
          
  console.log(filterPhotoList);

  this.setState({

      visiblePhotos:filterPhotoList

  })
    
}







	render(){

		return(
          
           <section className ="container-fluid">


           <div className ="row gap"> 
           <div className ="col-sm-4">
           <input 
           type ="text" 
           className ="form-control" 
           onChange ={this.filterPhotos}
           ref ={node =>(this.filterText = node)}

           /> </div>
            <div className ="col-sm-4"><button className ="btn btn-primary" onClick ={this.filterPhotos}>Filter</button></div>
            </div> 



           <div className ="row"> 
          
             {this.state.visiblePhotos.map((photo) =>{

                return <div key ={photo.id}className = "col-md-3">               

                <div className="card">
                 
                 <img src ={photo.thumbnailUrl} className ="card-img-top" alt ="" />

                <div className="card-body">
                <h4 className="card-title">{photo.title}</h4>
                 <p className="card-text">Some example text some example text. John Doe is an architect and engineer</p>
                 <NavLink to ={`/${photo.id}`} className="btn btn-primary">See Profile</NavLink>
                 </div>
                 
                 

                 </div>



               </div>

              





             }
             	)

             }
              </div>

           </section>


	   )
	}
}



export default MovieList;