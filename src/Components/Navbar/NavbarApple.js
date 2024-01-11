import React from "react";
import { NavLink as Link } from "react-router-dom";

import styled from 'styled-components';

const NavbarContainer = styled.ul`
  list-style-type: none;
  background-color: #8154af;
  overflow: hidden;
  margin: 0;
  padding: 0;
`;

const NavItem = styled.li`
  float: left;
`;

const NavLink = styled.a`
  display: block;
  color: white;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;

  &:hover {
    background-color: #8154af;
  }
`;

const ActiveLink = styled(NavLink)`
  background-color: #31174a;
`;

function NavbarApple() {
  return (
    <NavbarContainer>
      <NavItem>
        <ActiveLink href="/apple">Apple</ActiveLink>
      </NavItem>
      <NavItem>
        <NavLink href="/microsoft">Microsoft</NavLink>
      </NavItem>
      <NavItem>
        <NavLink href="/google">Google</NavLink>
      </NavItem>
      <NavItem>
        <NavLink href="/amazon">Amazon</NavLink>
      </NavItem> 
      <NavItem>
        <NavLink href="/nvidia">Nvidia</NavLink>
      </NavItem>
    </NavbarContainer>
  );
}

export default NavbarApple;


