import React, { useState } from 'react'
import './Home.css'
import {motion,useScroll,useTransform} from 'framer-motion';
import { Link } from 'react-router-dom';

function Home() {
    const [isOpen, setIsOpen] = useState(false);
    return (

        <>
        <div className='homecontainer'>
           <video src='heroo.mp4' autoPlay loop muted/>
            <div className='container'>
             <motion.h1 
                    className="neonText"  
                    initial={{x:"10rem",opacity:0}}
                    animate={{x:0,opacity:1}}
                    transition={{duration:2,type:'spring'}}>
                        Welcome to StockZ!!
             </motion.h1>

             <motion.h4 
             className="neonText"  
             initial={{x:"-10rem",opacity:0}}
             animate={{x:0,opacity:1}}
             transition={{duration:3,type:'spring'}}
             >Select a Company 
             </motion.h4>

             <div className='links'>
                <Link to='/apple'className='btn'>Apple</Link>
                <Link to='/microsoft' className='btn'>Microsoft</Link>
                <Link to='/google' className='btn'>Google</Link>
                <Link to='/amazon' className='btn'>Amazon</Link>
                <Link to='/nvidia' className='btn'>NVIDIA</Link>
             </div>
             
            </div>
        </div>
        
        </>
    )
}

export default Home
